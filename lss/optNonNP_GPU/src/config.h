/*
 * Northeastern University
 * Computer Architecture Research Group
 * NUCAR
 */

#ifndef CONFIG_H
#define CONFIG_H

/*
 * Configuration Options:
 * 	Variable controlling Masks and Bits (for efficient coding)
 * 	Allowed values:
 *		- THREAD_TILE_SIZE = 2. There's a bug for value of 1
		- BLOCK_TILE_SIZE = 32 | 16 | 8 | 4 | 2
 */
#define THREAD_TILE_SIZE 2
#define BLOCK_TILE_SIZE 16

// Defining verbosity of run
#define VERBOSE

// Defining important constants
#define MAX_LABELS_PER_IMAGE 32

// Threshold of max number of iterations of analysis
#define MAX_ITER 5000

// timing directives
#define CUDA_TIMING
#define KERNEL_TIMING

// Debug Cuda Errors
//#define DEBUG

// Output Brightness for inside object
#define OUTPUT_BRIGHTNESS 50

/*
 * Variable controlling Masks and Bits (for efficiency)
 *
 * Definitions as follows:
 * 	- BTSMask: 	BLOCK_TILE_SIZE-1
 * 	- BTSB: 	BLOCK_TILE_SIZE bits (2^BTSB = BLOCK_TILE_SIZE)
 * 	- TILE_SIZE: 	BLOCK_TILE_SIZE*THREAD_TILE_SIZE
 * 	- TSMask: 	TILE_SIZE-1
 * 	- TSB: 		TILE_SIZE bits (2^TSB = TILE_SIZE)
 *	- TTSMask:	THREAD_TILE_SIZE-1
 *	- TTSB:		THREAD_TILE_SIZE bits (2^TTSB = THREAD_TILE_SIZE)
 *
 */
/*#if BLOCK_TILE_SIZE >= 64
	#undef BLOCK_TILE_SIZE
	#define BLOCK_TILE_SIZE 16
	#define BTSMask		31
	#define BTSB		5

	// Due to shared memory space, TTS has to be 1
	#undef THREAD_TILE_SIZE
	#define THREAD_TILE_SIZE 1
	#define TILE_SIZE	64
	#define TSMask		31
	#define TSB		5
	#define TTSMask		0
	#define TTSB 		0

*/
//#elif BLOCK_TILE_SIZE >= 32
#if BLOCK_TILE_SIZE >= 32
	#undef BLOCK_TILE_SIZE
	#define BLOCK_TILE_SIZE 16
	#define BTSMask		31
	#define BTSB		5

	#if THREAD_TILE_SIZE >= 2
		#undef THREAD_TILE_SIZE
		#define THREAD_TILE_SIZE 2
		#define TILE_SIZE	64
		#define TSMask		63
		#define TSB		6
		#define TTSMask		1
		#define TTSB 		1
	/*#else
		#undef THREAD_TILE_SIZE
		#define THREAD_TILE_SIZE 1
		#define TILE_SIZE	32
		#define TSMask		31
		#define TSB		5
		#define TTSMask		0
		#define TTSB 		0*/
	#endif

#elif BLOCK_TILE_SIZE >= 16
	#define BTSMask		15
	#define BTSB		4

	#if THREAD_TILE_SIZE >= 2
		#undef THREAD_TILE_SIZE
		#define THREAD_TILE_SIZE 2
		#define TILE_SIZE	32
		#define TSMask		31
		#define TSB		5
		#define TTSMask		1
		#define TTSB 		1
	/*#else
		#undef THREAD_TILE_SIZE
		#define THREAD_TILE_SIZE 1
		#define TILE_SIZE	16
		#define TSMask		15
		#define TSB		4
		#define TTSMask		0
		#define TTSB 		0*/
	#endif

#elif BLOCK_TILE_SIZE >= 8
	#define BTSMask		7
	#define BTSB		3

	#if THREAD_TILE_SIZE >= 2
		#undef THREAD_TILE_SIZE
		#define THREAD_TILE_SIZE 2
		#define TILE_SIZE	16
		#define TSMask		15
		#define TSB		4
		#define TTSMask		1
		#define TTSB 		1
	/*#else
		#undef THREAD_TILE_SIZE
		#define THREAD_TILE_SIZE 1
		#define TILE_SIZE	8
		#define TSMask		7
		#define TSB		3
		#define TTSMask		0
		#define TTSB 		0*/
	#endif

#elif BLOCK_TILE_SIZE >= 4
	#define BTSMask		3
	#define BTSB		2

	#if THREAD_TILE_SIZE >= 2
		#undef THREAD_TILE_SIZE
		#define THREAD_TILE_SIZE 2
		#define TILE_SIZE	8
		#define TSMask		7
		#define TSB		3
		#define TTSMask		1
		#define TTSB 		1
	/*#else
		#undef THREAD_TILE_SIZE
		#define THREAD_TILE_SIZE 1
		#define TILE_SIZE	4
		#define TSMask		3
		#define TSB		2
		#define TTSMask		0
		#define TTSB 		0*/
	#endif

#elif BLOCK_TILE_SIZE >= 2
	#define BTSMask		1
	#define BTSB		1

	#if THREAD_TILE_SIZE >= 2
		#undef THREAD_TILE_SIZE
		#define THREAD_TILE_SIZE 2
		#define TILE_SIZE	4
		#define TSMask		3
		#define TSB		2
		#define TTSMask		1
		#define TTSB 		1
	/*#else
		#undef THREAD_TILE_SIZE
		#define THREAD_TILE_SIZE 1
		#define TILE_SIZE	2
		#define TSMask		1
		#define TSB		1
		#define TTSMask		0
		#define TTSB 		0*/
	#endif

#else // Default Jjust in case
	#undef BLOCK_TILE_SIZE
	#define BLOCK_TILE_SIZE 16
	#define BTSMask		15
	#define BTSB		4

	#if THREAD_TILE_SIZE >= 2
		#undef THREAD_TILE_SIZE
		#define THREAD_TILE_SIZE 2
		#define TILE_SIZE	32
		#define TSMask		31
		#define TSB		5
		#define TTSMask		1
		#define TTSB 		1
	/*#else
		#undef THREAD_TILE_SIZE
		#define THREAD_TILE_SIZE 1
		#define TILE_SIZE	16
		#define TSMask		15
		#define TSB		4
		#define TTSMask		0
		#define TTSB 		0*/
	#endif
#endif

#endif
