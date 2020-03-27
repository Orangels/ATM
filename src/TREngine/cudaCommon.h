#pragma once
#include <cuda_runtime_api.h>
#define exitIfCudaError(err)  __check ( (err), #err, __FILE__, __LINE__ )


template< typename T >
void __check(T result, char const *const func, const char *const file, int const line)
{
	if (result)
	{
		std::cout << "CUDA error at" << file << line << ", " << static_cast<unsigned int>(result) << __cudaGetErrorEnum(result) << func << std::endl;
		cudaDeviceReset();
		getchar();
		exit(EXIT_FAILURE);
	}
}

static const char *__cudaGetErrorEnum(cudaError_t error)
{
	switch (error)
	{
	case cudaSuccess:
		return "cudaSuccess";

	case cudaErrorMissingConfiguration:
		return "cudaErrorMissingConfiguration";

	case cudaErrorMemoryAllocation:
		return "cudaErrorMemoryAllocation";

	case cudaErrorInitializationError:
		return "cudaErrorInitializationError";

	case cudaErrorLaunchFailure:
		return "cudaErrorLaunchFailure";

	case cudaErrorPriorLaunchFailure:
		return "cudaErrorPriorLaunchFailure";

	case cudaErrorLaunchTimeout:
		return "cudaErrorLaunchTimeout";

	case cudaErrorLaunchOutOfResources:
		return "cudaErrorLaunchOutOfResources";

	case cudaErrorInvalidDeviceFunction:
		return "cudaErrorInvalidDeviceFunction";

	case cudaErrorInvalidConfiguration:
		return "cudaErrorInvalidConfiguration";

	case cudaErrorInvalidDevice:
		return "cudaErrorInvalidDevice";

	case cudaErrorInvalidValue:
		return "cudaErrorInvalidValue";

	case cudaErrorInvalidPitchValue:
		return "cudaErrorInvalidPitchValue";

	case cudaErrorInvalidSymbol:
		return "cudaErrorInvalidSymbol";

	case cudaErrorMapBufferObjectFailed:
		return "cudaErrorMapBufferObjectFailed";

	case cudaErrorUnmapBufferObjectFailed:
		return "cudaErrorUnmapBufferObjectFailed";

	case cudaErrorInvalidHostPointer:
		return "cudaErrorInvalidHostPointer";

	case cudaErrorInvalidDevicePointer:
		return "cudaErrorInvalidDevicePointer";

	case cudaErrorInvalidTexture:
		return "cudaErrorInvalidTexture";

	case cudaErrorInvalidTextureBinding:
		return "cudaErrorInvalidTextureBinding";

	case cudaErrorInvalidChannelDescriptor:
		return "cudaErrorInvalidChannelDescriptor";

	case cudaErrorInvalidMemcpyDirection:
		return "cudaErrorInvalidMemcpyDirection";

	case cudaErrorAddressOfConstant:
		return "cudaErrorAddressOfConstant";

	case cudaErrorTextureFetchFailed:
		return "cudaErrorTextureFetchFailed";

	case cudaErrorTextureNotBound:
		return "cudaErrorTextureNotBound";

	case cudaErrorSynchronizationError:
		return "cudaErrorSynchronizationError";

	case cudaErrorInvalidFilterSetting:
		return "cudaErrorInvalidFilterSetting";

	case cudaErrorInvalidNormSetting:
		return "cudaErrorInvalidNormSetting";

	case cudaErrorMixedDeviceExecution:
		return "cudaErrorMixedDeviceExecution";

	case cudaErrorCudartUnloading:
		return "cudaErrorCudartUnloading";

	case cudaErrorUnknown:
		return "cudaErrorUnknown";

	case cudaErrorNotYetImplemented:
		return "cudaErrorNotYetImplemented";

	case cudaErrorMemoryValueTooLarge:
		return "cudaErrorMemoryValueTooLarge";

	case cudaErrorInvalidResourceHandle:
		return "cudaErrorInvalidResourceHandle";

	case cudaErrorNotReady:
		return "cudaErrorNotReady";

	case cudaErrorInsufficientDriver:
		return "cudaErrorInsufficientDriver";

	case cudaErrorSetOnActiveProcess:
		return "cudaErrorSetOnActiveProcess";

	case cudaErrorInvalidSurface:
		return "cudaErrorInvalidSurface";

	case cudaErrorNoDevice:
		return "cudaErrorNoDevice";

	case cudaErrorECCUncorrectable:
		return "cudaErrorECCUncorrectable";

	case cudaErrorSharedObjectSymbolNotFound:
		return "cudaErrorSharedObjectSymbolNotFound";

	case cudaErrorSharedObjectInitFailed:
		return "cudaErrorSharedObjectInitFailed";

	case cudaErrorUnsupportedLimit:
		return "cudaErrorUnsupportedLimit";

	case cudaErrorDuplicateVariableName:
		return "cudaErrorDuplicateVariableName";

	case cudaErrorDuplicateTextureName:
		return "cudaErrorDuplicateTextureName";

	case cudaErrorDuplicateSurfaceName:
		return "cudaErrorDuplicateSurfaceName";

	case cudaErrorDevicesUnavailable:
		return "cudaErrorDevicesUnavailable";

	case cudaErrorInvalidKernelImage:
		return "cudaErrorInvalidKernelImage";

	case cudaErrorNoKernelImageForDevice:
		return "cudaErrorNoKernelImageForDevice";

	case cudaErrorIncompatibleDriverContext:
		return "cudaErrorIncompatibleDriverContext";

	case cudaErrorPeerAccessAlreadyEnabled:
		return "cudaErrorPeerAccessAlreadyEnabled";

	case cudaErrorPeerAccessNotEnabled:
		return "cudaErrorPeerAccessNotEnabled";

	case cudaErrorDeviceAlreadyInUse:
		return "cudaErrorDeviceAlreadyInUse";

	case cudaErrorProfilerDisabled:
		return "cudaErrorProfilerDisabled";

	case cudaErrorProfilerNotInitialized:
		return "cudaErrorProfilerNotInitialized";

	case cudaErrorProfilerAlreadyStarted:
		return "cudaErrorProfilerAlreadyStarted";

	case cudaErrorProfilerAlreadyStopped:
		return "cudaErrorProfilerAlreadyStopped";

		/* Since CUDA 4.0*/
	case cudaErrorAssert:
		return "cudaErrorAssert";

	case cudaErrorTooManyPeers:
		return "cudaErrorTooManyPeers";

	case cudaErrorHostMemoryAlreadyRegistered:
		return "cudaErrorHostMemoryAlreadyRegistered";

	case cudaErrorHostMemoryNotRegistered:
		return "cudaErrorHostMemoryNotRegistered";

		/* Since CUDA 5.0 */
	case cudaErrorOperatingSystem:
		return "cudaErrorOperatingSystem";

	case cudaErrorPeerAccessUnsupported:
		return "cudaErrorPeerAccessUnsupported";

	case cudaErrorLaunchMaxDepthExceeded:
		return "cudaErrorLaunchMaxDepthExceeded";

	case cudaErrorLaunchFileScopedTex:
		return "cudaErrorLaunchFileScopedTex";

	case cudaErrorLaunchFileScopedSurf:
		return "cudaErrorLaunchFileScopedSurf";

	case cudaErrorSyncDepthExceeded:
		return "cudaErrorSyncDepthExceeded";

	case cudaErrorLaunchPendingCountExceeded:
		return "cudaErrorLaunchPendingCountExceeded";

	case cudaErrorNotPermitted:
		return "cudaErrorNotPermitted";

	case cudaErrorNotSupported:
		return "cudaErrorNotSupported";

		/* Since CUDA 6.0 */
	case cudaErrorHardwareStackError:
		return "cudaErrorHardwareStackError";

	case cudaErrorIllegalInstruction:
		return "cudaErrorIllegalInstruction";

	case cudaErrorMisalignedAddress:
		return "cudaErrorMisalignedAddress";

	case cudaErrorInvalidAddressSpace:
		return "cudaErrorInvalidAddressSpace";

	case cudaErrorInvalidPc:
		return "cudaErrorInvalidPc";

	case cudaErrorIllegalAddress:
		return "cudaErrorIllegalAddress";

		/* Since CUDA 6.5*/
	case cudaErrorInvalidPtx:
		return "cudaErrorInvalidPtx";

	case cudaErrorInvalidGraphicsContext:
		return "cudaErrorInvalidGraphicsContext";

	case cudaErrorStartupFailure:
		return "cudaErrorStartupFailure";

	case cudaErrorApiFailureBase:
		return "cudaErrorApiFailureBase";

		/* Since CUDA 8.0*/
	case cudaErrorNvlinkUncorrectable:
		return "cudaErrorNvlinkUncorrectable";
	}

	return "<unknown>";
}