#include "Cafe/Savestate/SaveState.h"

#include "Cafe/CafeSystem.h"
#include "Cafe/HW/MMU/MMU.h"
#include "Cafe/OS/libs/coreinit/coreinit_Thread.h"
#include "Cafe/OS/libs/coreinit/coreinit_Scheduler.h"
#include "Cafe/OS/libs/TCL/TCL.h"
#include "Cafe/HW/Latte/Core/Latte.h"
#include "Cafe/HW/Latte/Core/LatteTiming.h"
#include "Cafe/HW/Latte/Core/LatteAsyncCommands.h"
#include "Cafe/HW/Espresso/Recompiler/PPCRecompiler.h"
#include "Cafe/IOSU/fsa/iosu_fsa.h"
#include "Cemu/Logging/CemuLogging.h"
#include "Cafe/OS/libs/coreinit/coreinit_Alarm.h"
#include "Cafe/OS/libs/snd_core/ax.h"
#include "Cafe/OS/libs/gx2/GX2_Event.h"
#include "Cafe/Cheats/CheatManager.h"

#include <chrono>
#include <ctime>
#include <fstream>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

// SysAllocator bump pointer, defined in coreinit_MEM.cpp. It is a host-side
// variable but points into the CEMU_AREA of emulated memory, so it must be
// captured and restored alongside the memory dump.
extern MPTR sysAreaAllocatorOffset;

namespace
{
		constexpr uint32 kStateMagic = 0x31475343; // "CSG1"
// version 2 adds a host-state blob after the FSA file entries (AX voice
// lists, AX callbacks and the coreinit alarm registry). Version 1 files
// (without the blob) can still be restored, the host lists are then left
// as they are
		constexpr uint32 kStateVersion = 2;

		constexpr uint32 kThreadsIdleTimeoutMs = 2000;
		constexpr uint32 kGpuIdleTimeoutMs = 2000;

		struct SaveStateHeader
		{
				uint32 magic;
				uint32 version;
				uint64 titleId;
				uint64 timestamp;
				uint32 rangeCount;
				uint32 sysAreaAllocatorOffset;
				uint32 fsEntryCount;
				uint32 reserved;
		};
		static_assert(sizeof(SaveStateHeader) == 40);

		struct RangeHeader
		{
				uint32 baseAddress;
				uint32 size;
		};
		static_assert(sizeof(RangeHeader) == 8);

		void LogInfo(std::string_view msg)
		{
				cemuLog_log(LogType::Force, "[SaveState] {}", msg);
		}

		void LogWarning(std::string_view msg)
		{
				cemuLog_log(LogType::Force, "[SaveState] WARNING: {}", msg);
		}

		// true when no emulated thread is still executing (all register contexts
		// have been stored back into the OSThread structures in emulated memory)
		bool AreThreadsIdle()
		{
				return coreinit::AreAllThreadContextsStored();
		}

		// true when the GPU has processed every submitted command buffer and no
		// buffer swap is pending, which is the safest point to capture memory
		bool IsAtSafeGpuPoint()
		{
				if (!TCL::IsGpuIdle())
						return false;
				if (LatteGPUState.sharedArea)
				{
						const uint32 requested = _swapEndianU32(LatteGPUState.sharedArea->flipRequestCountBE);
						const uint32 executed = _swapEndianU32(LatteGPUState.sharedArea->flipExecuteCountBE);
						if (requested != executed)
								return false;
				}
				return true;
		}
		// host-state blob section types
		constexpr uint32 kHostSectionAx = 1;
		constexpr uint32 kHostSectionAlarms = 2;

		void AppendU32LE(std::vector<uint8>& out, uint32 v)
		{
				out.push_back((uint8)(v));
				out.push_back((uint8)(v >> 8));
				out.push_back((uint8)(v >> 16));
				out.push_back((uint8)(v >> 24));
		}

		uint32 ReadU32LE(const uint8* data)
		{
				return (uint32)data[0] | ((uint32)data[1] << 8) | ((uint32)data[2] << 16) | ((uint32)data[3] << 24);
		}

		// captures the host-side state (AX voice lists/callbacks, alarm registry)
		// which is not covered by the emulated memory snapshot
		void AppendHostStateSections(std::vector<uint8>& blob)
		{
				std::vector<uint8> section;
				snd_core::SavestateCaptureHostState(section);
				AppendU32LE(blob, kHostSectionAx);
				AppendU32LE(blob, (uint32)section.size());
				blob.insert(blob.end(), section.begin(), section.end());
				section.clear();
				coreinit::SavestateCaptureAlarms(section);
				AppendU32LE(blob, kHostSectionAlarms);
				AppendU32LE(blob, (uint32)section.size());
				blob.insert(blob.end(), section.begin(), section.end());
		}
}

namespace CafeSaveState
{
		EmulationFreeze::~EmulationFreeze()
		{
				end();
		}

		bool EmulationFreeze::begin(std::string& errorOut, bool gpuIdleRequired)
		{
				if (m_active)
						return true;

				if (!CafeSystem::IsTitleRunning())
				{
						errorOut = "No title is running";
						return false;
				}

				const auto freezeStart = std::chrono::steady_clock::now();
				auto elapsedMs = [&freezeStart]() { return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - freezeStart).count(); };
				m_wasPaused = CafeSystem::IsTitlePaused();
				if (!m_wasPaused)
				{
						CafeSystem::PauseTitle();
						m_pausedByUs = true;
				}

				// wait until every suspended PPC thread has finished its current
				// timeslice and stored its registers into emulated memory
				const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(kThreadsIdleTimeoutMs);
				while (std::chrono::steady_clock::now() < deadline)
				{
						if (AreThreadsIdle())
								break;
						std::this_thread::sleep_for(std::chrono::milliseconds(1));
				}
				if (!AreThreadsIdle())
				{
						errorOut = "Emulated threads did not become idle in time";
						if (m_pausedByUs)
						{
								CafeSystem::ResumeTitle();
								m_pausedByUs = false;
						}
						return false;
				}

				LogInfo(fmt::format("freeze: threads idle ({} ms)", elapsedMs()));
				// drain the command ring before pausing the vsync timing - a GPU thread
				// blocked in a flip wait needs the vsync counter to keep advancing
				if (!TCL::WaitForGPUIdle(kGpuIdleTimeoutMs))
				{
								if (gpuIdleRequired)
								{
												errorOut = "GPU did not become idle in time";
												if (m_pausedByUs)
												{
																CafeSystem::ResumeTitle();
																m_pausedByUs = false;
												}
												return false;
								}
								LogWarning("GPU did not retire all commands within the timeout");
				}

				// commit all pending GPU work (occlusion queries, texture readbacks)
				// into emulated memory now that the command ring is drained
				LatteAsyncCommands_queueSaveStateGpuSync(false);
				if (!LatteAsyncCommands_waitUntilAllProcessedWithTimeout(kGpuIdleTimeoutMs))
				{
						LogWarning("GPU thread did not process the pre-capture sync in time");
						if (m_pausedByUs)
						{
								CafeSystem::ResumeTitle();
								m_pausedByUs = false;
						}
						errorOut = "GPU sync timed out";
						return false;
				}

				LogInfo(fmt::format("freeze: gpu drained + sync done ({} ms)", elapsedMs()));
				// stop the virtual vsync timer and the OS system events (AX/alarm/NFP)
				// so that no host thread mutates emulated memory while we work
				LatteTiming_setPaused(true);
				coreinit::SetSystemEventsPaused(true);
				m_frozenTiming = true;

				// keep the GPU thread parked while the caller overwrites emulated memory
				// (also stops OSScreen uploads from racing the snapshot)
				g_savestateFreezeGpuThread.store(true);

				if (!IsAtSafeGpuPoint())
				{
								// not fatal: the game will simply redraw the affected surfaces
								LogWarning("Capture point is not an exact frame boundary");
				}

				LogInfo(fmt::format("freeze: active ({} ms)", elapsedMs()));
				m_active = true;
				return true;
		}

		void EmulationFreeze::end()
		{
				// let the GPU thread continue first so it can process the cache reset
				g_savestateFreezeGpuThread.store(false);
				if (m_resetGpuCaches)
				{
								m_resetGpuCaches = false;
								// drop all cached GPU data so everything is re-uploaded from the
								// restored emulated memory
								LatteAsyncCommands_queueSaveStateGpuSync(true);
								if (!LatteAsyncCommands_waitUntilAllProcessedWithTimeout(kGpuIdleTimeoutMs))
										LogWarning("GPU cache reset did not complete in time - continuing");
				}
				LogInfo("freeze: releasing timing + gpu park");
				if (m_frozenTiming)
				{
						coreinit::SetSystemEventsPaused(false);
						LatteTiming_setPaused(false);
						m_frozenTiming = false;
				}
				if (m_pausedByUs)
				{
						CafeSystem::ResumeTitle();
						m_pausedByUs = false;
				}
				LogInfo("freeze: emulation resumed");
				m_active = false;
		}

		void EmulationFreeze::RequestGpuCacheReset()
		{
				m_resetGpuCaches = true;
		}

		OperationResult CaptureStateToFileLocked(uint64 titleId, const std::filesystem::path& filePath)
		{
				OperationResult result;

				std::error_code ec;
				fs::path tmpPath = filePath;
				tmpPath += ".tmp";
				fs::create_directories(filePath.parent_path(), ec);

				std::ofstream out(tmpPath, std::ios::binary | std::ios::trunc);
				if (!out)
				{
						result.errorMessage = "Failed to create state file";
						return result;
				}

				// collect all mapped memory ranges
				std::vector<MMURange*> ranges;
				for (MMURange* range : memory_getMMURanges())
				{
						if (range->isMapped() && range->getPtr() && range->getSize() > 0)
								ranges.emplace_back(range);
				}

				// collect the state of all files the game currently holds open
				std::vector<iosu::fsa::FSAFileStateEntry> fileStates;
				iosu::fsa::FSACollectOpenFileStates(fileStates);

				SaveStateHeader header{};
				header.magic = kStateMagic;
				header.version = kStateVersion;
				header.titleId = titleId;
				header.timestamp = (uint64)time(nullptr);
				header.rangeCount = (uint32)ranges.size();
				header.sysAreaAllocatorOffset = (uint32)sysAreaAllocatorOffset;
				header.fsEntryCount = (uint32)fileStates.size();
				header.reserved = 0;

				out.write((const char*)&header, sizeof(header));

				uint64 totalBytes = sizeof(header);
				for (const MMURange* range : ranges)
				{
						RangeHeader rangeHeader{ range->getBase(), range->getSize() };
						out.write((const char*)&rangeHeader, sizeof(rangeHeader));
						out.write((const char*)range->getPtr(), range->getSize());
						if (!out)
						{
								out.close();
								fs::remove(tmpPath, ec);
								result.errorMessage = "Failed to write memory range to state file";
								LogWarning(result.errorMessage);
								return result;
						}
						totalBytes += sizeof(rangeHeader) + range->getSize();
				}

				for (const iosu::fsa::FSAFileStateEntry& entry : fileStates)
				{
						const uint32 pathLength = (uint32)entry.path.size();
						out.write((const char*)&pathLength, sizeof(pathLength));
						out.write(entry.path.data(), pathLength);
						out.write((const char*)&entry.seek, sizeof(entry.seek));
						out.write((const char*)&entry.fileSize, sizeof(entry.fileSize));
						const uint8 writable = entry.writable ? 1 : 0;
						out.write((const char*)&writable, sizeof(writable));
						const uint32 contentLength = (uint32)entry.content.size();
						out.write((const char*)&contentLength, sizeof(contentLength));
						if (contentLength > 0)
								out.write((const char*)entry.content.data(), contentLength);
						if (!out)
						{
								out.close();
								fs::remove(tmpPath, ec);
								result.errorMessage = "Failed to write file handle state";
								LogWarning(result.errorMessage);
								return result;
						}
						totalBytes += contentLength;
				}

				// host-side state (AX voice lists/callbacks, alarm registry) which is
				// not part of emulated memory
				std::vector<uint8> hostBlob;
				AppendHostStateSections(hostBlob);
				const uint32 hostBlobSize = (uint32)hostBlob.size();
				out.write((const char*)&hostBlobSize, sizeof(hostBlobSize));
				if (hostBlobSize > 0)
								out.write((const char*)hostBlob.data(), (std::streamsize)hostBlob.size());
				totalBytes += sizeof(hostBlobSize) + hostBlob.size();

				out.close();
				if (!out)
{
								fs::remove(tmpPath, ec);
								result.errorMessage = "Failed to flush state file";
								return result;
				}

				// move the finished snapshot into place (removes any old state file)
				fs::remove(filePath, ec);
				fs::rename(tmpPath, filePath, ec);
				if (ec)
				{
						fs::remove(tmpPath, ec);
						result.errorMessage = "Failed to finalize state file";
						LogWarning(result.errorMessage);
						return result;
				}

				LogInfo(fmt::format("Captured state for {:016x} ({:.1f} MB, {} file handles)",
						titleId, (double)totalBytes / 1024.0 / 1024.0, fileStates.size()));

				result.success = true;
				return result;
		}

		OperationResult RestoreStateFromFileLocked(uint64 expectedTitleId, const std::filesystem::path& filePath)
		{
				OperationResult result;
				std::error_code ec;
				if (!fs::exists(filePath, ec))
				{
						result.errorMessage = "State file does not exist";
						return result;
				}

				// parse and validate the whole file BEFORE touching any emulation state.
				// the memory payloads are kept as stream offsets so we never need to
				// hold the entire 1.4+ GB snapshot in host memory
				std::ifstream in(filePath, std::ios::binary);
				if (!in)
				{
						result.errorMessage = "Failed to open state file";
						return result;
				}

				SaveStateHeader header{};
				in.read((char*)&header, sizeof(header));
				if (!in || header.magic != kStateMagic || (header.version != 1 && header.version != 2))
				{
						result.errorMessage = "Not a valid savestate file";
						return result;
				}
				if (header.titleId != expectedTitleId)
				{
						result.errorMessage = fmt::format("State file belongs to title {:016x}", header.titleId);
						return result;
				}

				struct PendingRange
				{
						uint32 baseAddress;
						uint32 size;
						uint64 fileOffset;
				};
				std::vector<PendingRange> pendingRanges;
				pendingRanges.reserve(header.rangeCount);

				// resolve and validate every range against the current memory mapping
				const std::vector<MMURange*> currentRanges = memory_getMMURanges();
				for (uint32 i = 0; i < header.rangeCount; ++i)
				{
						RangeHeader rangeHeader{};
						in.read((char*)&rangeHeader, sizeof(rangeHeader));
						if (!in)
						{
								result.errorMessage = "State file is truncated (range table)";
								return result;
						}

						MMURange* target = nullptr;
						for (MMURange* range : currentRanges)
						{
								if (range->getBase() == rangeHeader.baseAddress)
								{
										target = range;
										break;
								}
						}
						if (!target || !target->isMapped() || !target->getPtr())
						{
								result.errorMessage = fmt::format("Memory range {:08x} is not mapped anymore", rangeHeader.baseAddress);
								return result;
						}
						if (target->getSize() != rangeHeader.size)
						{
								result.errorMessage = fmt::format("Size of memory range {:08x} changed", rangeHeader.baseAddress);
								return result;
						}

						PendingRange pending;
						pending.baseAddress = rangeHeader.baseAddress;
						pending.size = rangeHeader.size;
						pending.fileOffset = (uint64)in.tellg();
						pendingRanges.emplace_back(pending);
						in.seekg((std::streamoff)pending.size, std::ios::cur);
				}

				// read the file handle states. all memory payload reads in the
				// apply phase below are guaranteed to succeed because the
				// validation loop above read through every range payload
				std::vector<iosu::fsa::FSAFileStateEntry> fileStates;
				for (uint32 i = 0; i < header.fsEntryCount; ++i)
				{
						iosu::fsa::FSAFileStateEntry entry;
						uint32 pathLength = 0;
						in.read((char*)&pathLength, sizeof(pathLength));
						if (!in || pathLength > 4096)
						{
								result.errorMessage = "State file is truncated (file table)";
								return result;
						}
						entry.path.resize(pathLength);
						in.read(entry.path.data(), pathLength);
						in.read((char*)&entry.seek, sizeof(entry.seek));
						in.read((char*)&entry.fileSize, sizeof(entry.fileSize));
						uint8 writable = 0;
						in.read((char*)&writable, sizeof(writable));
						entry.writable = writable != 0;
						uint32 contentLength = 0;
						in.read((char*)&contentLength, sizeof(contentLength));
						if (!in || contentLength > (64u * 1024u * 1024u))
						{
								result.errorMessage = "State file is truncated (file content)";
								return result;
						}
						entry.content.resize(contentLength);
						if (contentLength > 0)
								in.read((char*)entry.content.data(), contentLength);
						if (!in)
						{
								result.errorMessage = "State file is truncated (file content)";
								return result;
						}
						fileStates.emplace_back(std::move(entry));
				}

				// parse the host-state blob (version 2+). this happens before anything
				// is applied so a truncated file never touches the running emulation
				std::vector<std::pair<uint32, std::vector<uint8>>> hostSections;
				if (header.version >= 2)
				{
								uint32 hostBlobSize = 0;
								in.read((char*)&hostBlobSize, sizeof(hostBlobSize));
								if (!in || hostBlobSize > (4u * 1024u * 1024u))
								{
												result.errorMessage = "State file is truncated (host state)";
												return result;
								}
								std::vector<uint8> hostBlob(hostBlobSize);
								if (hostBlobSize > 0)
								{
												in.read((char*)hostBlob.data(), (std::streamsize)hostBlob.size());
												if (!in)
												{
																result.errorMessage = "State file is truncated (host state)";
																return result;
												}
								}
								size_t blobOffset = 0;
								while (blobOffset + 8 <= hostBlob.size())
								{
												const uint32 sectionType = ReadU32LE(hostBlob.data() + blobOffset);
												const uint32 sectionSize = ReadU32LE(hostBlob.data() + blobOffset + 4);
												blobOffset += 8;
												if (blobOffset + sectionSize > hostBlob.size())
																break;
												hostSections.emplace_back(sectionType, std::vector<uint8>(hostBlob.begin() + blobOffset, hostBlob.begin() + blobOffset + sectionSize));
												blobOffset += sectionSize;
								}
				}

				LogInfo(fmt::format("restore: validation done ({} ranges, {} file handles)", pendingRanges.size(), fileStates.size()));
				// everything validated - now apply the snapshot (caller holds the freeze)

				for (const PendingRange& pending : pendingRanges)
				{
						MMURange* target = nullptr;
						for (MMURange* range : currentRanges)
						{
								if (range->getBase() == pending.baseAddress)
								{
										target = range;
										break;
								}
						}
						in.seekg((std::streamoff)pending.fileOffset, std::ios::beg);
						in.read((char*)target->getPtr(), pending.size);
						if (!in)
						{
								result.errorMessage = "Failed to read memory range from state file";
								LogWarning(result.errorMessage);
								return result;
						}
				}

				// restore host-side state
				sysAreaAllocatorOffset = (MPTR)header.sysAreaAllocatorOffset;
				iosu::fsa::FSARestoreFileStates(fileStates);

				// restore the host-side registries which are not covered by the memory
				// snapshot (AX voice lists/callbacks, alarm registry)
				for (const auto& section : hostSections)
				{
								const uint8* sectionData = section.second.data();
								const uint8* sectionEnd = sectionData + section.second.size();
								if (section.first == kHostSectionAx)
								{
												if (!snd_core::SavestateRestoreHostState(sectionData, sectionEnd))
																LogWarning("AX host state section is invalid");
								}
								else if (section.first == kHostSectionAlarms)
								{
												if (!coreinit::SavestateRestoreAlarms(sectionData, sectionEnd))
																LogWarning("Alarm host state section is invalid");
								}
				}

				LogInfo("restore: memory + host state applied");
				// resync the host-side pending flip counter with the restored emulated
				// counters, otherwise the GPU keeps executing phantom flips and
				// GX2WaitSwap never observes request == execute
				if (LatteGPUState.sharedArea)
				{
								const uint32 requestedFlips = _swapEndianU32(LatteGPUState.sharedArea->flipRequestCountBE);
								const uint32 executedFlips = _swapEndianU32(LatteGPUState.sharedArea->flipExecuteCountBE);
								LatteGPUState.flipRequestCount.store(requestedFlips > executedFlips ? (uint64)(requestedFlips - executedFlips) : 0ull);
				}

				// drop GX2 event callbacks which were queued after the snapshot was
				// taken (they belong to the discarded timeline)
				GX2::ResetQueuedEventCallbacks();

				// rebuild the host scheduler state from the restored emulated memory
				coreinit::SavestateRebuildSchedulerAfterRestore();

				// the restored memory may contain different code than the currently
				// JIT-compiled fragments (self-modifying code), so drop everything
				PPCRecompiler_invalidateRange(0x00010000, 0x10000000);
				PPCRecompiler_invalidateRange(0xF6000000, 0xF8000000);

				LogInfo("restore: scheduler rebuilt + recompiler invalidated");
				// the host audio output must be reset after an in-place restore:
				// stale queued blocks, partially accumulated DMA groups and a
				// desynced frame gate would otherwise cause probabilistic silence
				snd_core::AXOut_ResetAfterRestore();

				// cheat patches live in emulated memory and were just overwritten
				// by the snapshot - re-apply all enabled cheats
				Cheats::CheatManager::GetInstance().ReapplyAll();

				LogInfo("restore: audio reset + cheats reapplied");
				LogInfo(fmt::format("Restored state for {:016x} ({} ranges, {} file handles)",
						expectedTitleId, pendingRanges.size(), fileStates.size()));

				result.success = true;
				return result;
		}

		OperationResult CaptureStateToFile(uint64 titleId, const std::filesystem::path& filePath)
		{
				EmulationFreeze freeze;
				std::string freezeError;
				if (!freeze.begin(freezeError))
					return { false, freezeError };
				return CaptureStateToFileLocked(titleId, filePath);
		}

		OperationResult RestoreStateFromFile(uint64 expectedTitleId, const std::filesystem::path& filePath)
		{
				// the file is validated before the freeze is acquired so that an
				// invalid file never interrupts the running emulation
				std::error_code ec;
				if (!std::filesystem::exists(filePath, ec))
					return { false, "State file does not exist" };
				EmulationFreeze freeze;
				std::string freezeError;
				if (!freeze.begin(freezeError))
								return { false, freezeError };
				auto result = RestoreStateFromFileLocked(expectedTitleId, filePath);
				if (result.success)
								freeze.RequestGpuCacheReset();
				return result;
		}
}
