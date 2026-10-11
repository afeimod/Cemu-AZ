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
		constexpr uint32 kStateVersion = 1;

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
}

namespace CafeSaveState
{
		EmulationFreeze::~EmulationFreeze()
		{
				end();
		}

		bool EmulationFreeze::begin(std::string& errorOut)
		{
				if (m_active)
						return true;

				if (!CafeSystem::IsTitleRunning())
				{
						errorOut = "No title is running";
						return false;
				}

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

				// stop the virtual vsync timer and the OS system events (AX/alarm/NFP)
				// so that no host thread mutates emulated memory while we work
				LatteTiming_setPaused(true);
				coreinit::SetSystemEventsPaused(true);
				m_frozenTiming = true;

				// let the GPU drain everything that was submitted before the pause
				if (!TCL::WaitForGPUIdle(kGpuIdleTimeoutMs))
						LogWarning("GPU did not retire all commands within the timeout");

				LatteAsyncCommands_waitUntilAllProcessed();

				if (!IsAtSafeGpuPoint())
				{
						// not fatal: the game will simply redraw the affected surfaces
						LogWarning("Capture point is not an exact frame boundary");
				}

				m_active = true;
				return true;
		}

		void EmulationFreeze::end()
		{
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
				m_active = false;
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
				if (!in || header.magic != kStateMagic || header.version != kStateVersion)
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

				// rebuild the host scheduler state from the restored emulated memory
				coreinit::SavestateRebuildSchedulerAfterRestore();

				// the restored memory may contain different code than the currently
				// JIT-compiled fragments (self-modifying code), so drop everything
				PPCRecompiler_invalidateRange(0x00010000, 0x10000000);
				PPCRecompiler_invalidateRange(0xF6000000, 0xF8000000);

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
				return RestoreStateFromFileLocked(expectedTitleId, filePath);
		}
}
