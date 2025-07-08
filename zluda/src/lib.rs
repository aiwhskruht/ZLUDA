use cuda_types::cuda::CUerror;
use std::sync::atomic::{AtomicBool, Ordering};
pub(crate) mod r#impl;

static INITIALIZED: AtomicBool = AtomicBool::new(true);
pub(crate) fn check_initialized() -> CUresult {
    match INITIALIZED.load(Ordering::SeqCst) {
        true => Ok(()),
        false => Err(CUerror::DEINITIALIZED)
    }
}
#[dtor::dtor]
fn deinitialize() {
    INITIALIZED.store(false, Ordering::SeqCst);
}

macro_rules! unimplemented {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
        $(
            #[cfg_attr(not(test), no_mangle)]
            #[allow(improper_ctypes)]
            #[allow(improper_ctypes_definitions)]
            pub unsafe extern $abi fn $fn_name ( $( $arg_id : $arg_type),* ) -> $ret_type {
                crate::r#impl::unimplemented()
            }
        )*
    };
}

macro_rules! implemented {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
        $(
            #[cfg_attr(not(test), no_mangle)]
            #[allow(improper_ctypes)]
            #[allow(improper_ctypes_definitions)]
            pub unsafe extern $abi fn $fn_name ( $( $arg_id : $arg_type),* ) -> $ret_type {
                if !initialized() {
                    return Err(CUerror::DEINITIALIZED);
                }
                cuda_base::cuda_normalize_fn!( crate::r#impl::$fn_name ) ($(crate::r#impl::FromCuda::from_cuda(&$arg_id)?),*)?;
                Ok(())
            }
        )*
    };
}

macro_rules! implemented_in_function {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
        $(
            #[cfg_attr(not(test), no_mangle)]
            #[allow(improper_ctypes)]
            #[allow(improper_ctypes_definitions)]
            pub unsafe extern $abi fn $fn_name ( $( $arg_id : $arg_type),* ) -> $ret_type {
                if !initialized() {
                    return Err(CUerror::DEINITIALIZED);
                }
                cuda_base::cuda_normalize_fn!( crate::r#impl::function::$fn_name ) ($(crate::r#impl::FromCuda::from_cuda(&$arg_id)?),*)?;
                Ok(())
            }
        )*
    };
}

cuda_base::cuda_function_declarations!(
    unimplemented,
    implemented <= [
        cuCtxCreate,
        cuCtxCreate_v2,
        cuCtxDestroy,
        cuCtxDestroy_v2,
        cuCtxPushCurrent,
        cuCtxPushCurrent_v2,
        cuCtxPopCurrent,
        cuCtxPopCurrent_v2,
        cuCtxSetCurrent,
        cuCtxGetCurrent,
        cuCtxGetDevice,
        cuCtxGetLimit,
        cuCtxSetLimit,
        cuCtxSetFlags,
        cuCtxGetStreamPriorityRange,
        cuCtxSynchronize,
        cuCtxSetCacheConfig,
        cuCtxGetApiVersion,
        cuDeviceComputeCapability,
        cuDeviceGet,
        cuDeviceGetAttribute,
        cuDeviceGetCount,
        cuDeviceGetLuid,
        cuDeviceGetName,
        cuDeviceGetProperties,
        cuDeviceGetUuid,
        cuDeviceGetUuid_v2,
        cuDevicePrimaryCtxRetain,
        cuDevicePrimaryCtxRelease,
        cuDevicePrimaryCtxRelease_v2,
        cuDevicePrimaryCtxReset,
        cuDevicePrimaryCtxReset_v2,
        cuDevicePrimaryCtxSetFlags,
        cuDevicePrimaryCtxSetFlags_v2,
        cuDevicePrimaryCtxGetState,
        cuDeviceTotalMem_v2,
        cuDriverGetVersion,
        cuFuncGetAttribute,
        cuGetExportTable,
        cuGetProcAddress,
        cuGetProcAddress_v2,
        cuInit,
        cuLibraryLoadData,
        cuLibraryGetModule,
        cuLibraryUnload,
        cuMemAlloc_v2,
        cuMemFree_v2,
        cuMemHostAlloc,
        cuMemFreeHost,
        cuMemGetAddressRange_v2,
        cuMemGetInfo_v2,
        cuMemcpyDtoH_v2,
        cuMemcpyHtoD_v2,
        cuMemsetD32_v2,
        cuMemsetD8_v2,
        cuModuleGetFunction,
        cuModuleGetLoadingMode,
        cuModuleLoadData,
        cuModuleUnload,
        cuPointerGetAttribute,
        cuStreamSynchronize,
        cuProfilerStart,
        cuProfilerStop,
    ],
    implemented_in_function <= [
        cuLaunchKernel,
    ]
);
