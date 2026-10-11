#pragma once

#include <jni.h>

// Quick save / quick load (save data snapshot slots with screenshot thumbnails)
// for the Android port. All functions operate on the currently running title.
namespace NativeQuickSave
{
	// Result codes shared between C++ and Kotlin
	enum QuickSaveResult : jint
	{
		QUICKSAVE_OK = 0,
		QUICKSAVE_ERROR_NOT_RUNNING = 1,   // no title is currently running
		QUICKSAVE_ERROR_NO_SAVE_DATA = 2,  // the game has not created any save data yet
		QUICKSAVE_ERROR_SLOT_EMPTY = 3,    // restore target slot does not exist
		QUICKSAVE_ERROR_IO = 4,            // filesystem error while copying
	};
}
