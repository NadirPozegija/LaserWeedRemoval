./demo.o : ../demo.cu \
    /usr/include/stdc-predef.h \
    /usr/local/cuda/include/cuda_runtime.h \
    /usr/local/cuda/include/host_config.h \
    /usr/include/features.h \
    /usr/include/arm-linux-gnueabihf/sys/cdefs.h \
    /usr/include/arm-linux-gnueabihf/bits/wordsize.h \
    /usr/include/arm-linux-gnueabihf/gnu/stubs.h \
    /usr/include/arm-linux-gnueabihf/gnu/stubs-hard.h \
    /usr/local/cuda/include/builtin_types.h \
    /usr/local/cuda/include/device_types.h \
    /usr/local/cuda/include/host_defines.h \
    /usr/local/cuda/include/driver_types.h \
    /usr/lib/gcc/arm-linux-gnueabihf/4.8/include-fixed/limits.h \
    /usr/lib/gcc/arm-linux-gnueabihf/4.8/include-fixed/syslimits.h \
    /usr/include/limits.h \
    /usr/include/arm-linux-gnueabihf/bits/posix1_lim.h \
    /usr/include/arm-linux-gnueabihf/bits/local_lim.h \
    /usr/include/linux/limits.h \
    /usr/include/arm-linux-gnueabihf/bits/posix2_lim.h \
    /usr/include/arm-linux-gnueabihf/bits/xopen_lim.h \
    /usr/include/arm-linux-gnueabihf/bits/stdio_lim.h \
    /usr/lib/gcc/arm-linux-gnueabihf/4.8/include/stddef.h \
    /usr/local/cuda/include/surface_types.h \
    /usr/local/cuda/include/texture_types.h \
    /usr/local/cuda/include/vector_types.h \
    /usr/local/cuda/include/channel_descriptor.h \
    /usr/local/cuda/include/cuda_runtime_api.h \
    /usr/local/cuda/include/cuda_device_runtime_api.h \
    /usr/local/cuda/include/driver_functions.h \
    /usr/local/cuda/include/vector_functions.h \
    /usr/local/cuda/include/common_functions.h \
    /usr/include/string.h \
    /usr/include/xlocale.h \
    /usr/include/time.h \
    /usr/include/arm-linux-gnueabihf/bits/time.h \
    /usr/include/arm-linux-gnueabihf/bits/types.h \
    /usr/include/arm-linux-gnueabihf/bits/typesizes.h \
    /usr/include/arm-linux-gnueabihf/bits/timex.h \
    /usr/include/c++/4.8/new \
    /usr/include/arm-linux-gnueabihf/c++/4.8/bits/c++config.h \
    /usr/include/arm-linux-gnueabihf/c++/4.8/bits/os_defines.h \
    /usr/include/arm-linux-gnueabihf/c++/4.8/bits/cpu_defines.h \
    /usr/include/c++/4.8/exception \
    /usr/include/c++/4.8/bits/atomic_lockfree_defines.h \
    /usr/include/stdio.h \
    /usr/include/libio.h \
    /usr/include/_G_config.h \
    /usr/include/wchar.h \
    /usr/lib/gcc/arm-linux-gnueabihf/4.8/include/stdarg.h \
    /usr/include/arm-linux-gnueabihf/bits/sys_errlist.h \
    /usr/include/stdlib.h \
    /usr/include/arm-linux-gnueabihf/bits/waitflags.h \
    /usr/include/arm-linux-gnueabihf/bits/waitstatus.h \
    /usr/include/endian.h \
    /usr/include/arm-linux-gnueabihf/bits/endian.h \
    /usr/include/arm-linux-gnueabihf/bits/byteswap.h \
    /usr/include/arm-linux-gnueabihf/bits/byteswap-16.h \
    /usr/include/arm-linux-gnueabihf/sys/types.h \
    /usr/include/arm-linux-gnueabihf/sys/select.h \
    /usr/include/arm-linux-gnueabihf/bits/select.h \
    /usr/include/arm-linux-gnueabihf/bits/sigset.h \
    /usr/include/arm-linux-gnueabihf/sys/sysmacros.h \
    /usr/include/arm-linux-gnueabihf/bits/pthreadtypes.h \
    /usr/include/alloca.h \
    /usr/include/arm-linux-gnueabihf/bits/stdlib-float.h \
    /usr/include/assert.h \
    /usr/local/cuda/include/math_functions.h \
    /usr/include/math.h \
    /usr/include/arm-linux-gnueabihf/bits/math-vector.h \
    /usr/include/arm-linux-gnueabihf/bits/libm-simd-decl-stubs.h \
    /usr/include/arm-linux-gnueabihf/bits/huge_val.h \
    /usr/include/arm-linux-gnueabihf/bits/huge_valf.h \
    /usr/include/arm-linux-gnueabihf/bits/huge_vall.h \
    /usr/include/arm-linux-gnueabihf/bits/inf.h \
    /usr/include/arm-linux-gnueabihf/bits/nan.h \
    /usr/include/arm-linux-gnueabihf/bits/mathdef.h \
    /usr/include/arm-linux-gnueabihf/bits/mathcalls.h \
    /usr/include/c++/4.8/cmath \
    /usr/include/c++/4.8/bits/cpp_type_traits.h \
    /usr/include/c++/4.8/ext/type_traits.h \
    /usr/include/c++/4.8/cstdlib \
    /usr/local/cuda/include/math_functions_dbl_ptx3.h \
    /usr/local/cuda/include/cuda_surface_types.h \
    /usr/local/cuda/include/cuda_texture_types.h \
    /usr/local/cuda/include/device_functions.h \
    /usr/local/cuda/include/sm_11_atomic_functions.h \
    /usr/local/cuda/include/sm_12_atomic_functions.h \
    /usr/local/cuda/include/sm_13_double_functions.h \
    /usr/local/cuda/include/sm_20_atomic_functions.h \
    /usr/local/cuda/include/sm_32_atomic_functions.h \
    /usr/local/cuda/include/sm_35_atomic_functions.h \
    /usr/local/cuda/include/sm_20_intrinsics.h \
    /usr/local/cuda/include/sm_30_intrinsics.h \
    /usr/local/cuda/include/sm_32_intrinsics.h \
    /usr/local/cuda/include/sm_35_intrinsics.h \
    /usr/local/cuda/include/surface_functions.h \
    /usr/local/cuda/include/texture_fetch_functions.h \
    /usr/local/cuda/include/texture_indirect_functions.h \
    /usr/local/cuda/include/surface_indirect_functions.h \
    /usr/local/cuda/include/device_launch_parameters.h \
    /usr/include/c++/4.8/iostream \
    /usr/include/c++/4.8/ostream \
    /usr/include/c++/4.8/ios \
    /usr/include/c++/4.8/iosfwd \
    /usr/include/c++/4.8/bits/stringfwd.h \
    /usr/include/c++/4.8/bits/memoryfwd.h \
    /usr/include/c++/4.8/bits/postypes.h \
    /usr/include/c++/4.8/cwchar \
    /usr/include/arm-linux-gnueabihf/bits/wchar.h \
    /usr/include/c++/4.8/bits/char_traits.h \
    /usr/include/c++/4.8/bits/stl_algobase.h \
    /usr/include/c++/4.8/bits/functexcept.h \
    /usr/include/c++/4.8/bits/exception_defines.h \
    /usr/include/c++/4.8/ext/numeric_traits.h \
    /usr/include/c++/4.8/bits/stl_pair.h \
    /usr/include/c++/4.8/bits/move.h \
    /usr/include/c++/4.8/bits/concept_check.h \
    /usr/include/c++/4.8/bits/stl_iterator_base_types.h \
    /usr/include/c++/4.8/bits/stl_iterator_base_funcs.h \
    /usr/include/c++/4.8/debug/debug.h \
    /usr/include/c++/4.8/bits/stl_iterator.h \
    /usr/include/c++/4.8/bits/localefwd.h \
    /usr/include/arm-linux-gnueabihf/c++/4.8/bits/c++locale.h \
    /usr/include/c++/4.8/clocale \
    /usr/include/locale.h \
    /usr/include/arm-linux-gnueabihf/bits/locale.h \
    /usr/include/c++/4.8/cctype \
    /usr/include/ctype.h \
    /usr/include/c++/4.8/bits/ios_base.h \
    /usr/include/c++/4.8/ext/atomicity.h \
    /usr/include/arm-linux-gnueabihf/c++/4.8/bits/gthr.h \
    /usr/include/arm-linux-gnueabihf/c++/4.8/bits/gthr-default.h \
    /usr/include/pthread.h \
    /usr/include/sched.h \
    /usr/include/arm-linux-gnueabihf/bits/sched.h \
    /usr/include/arm-linux-gnueabihf/bits/setjmp.h \
    /usr/include/arm-linux-gnueabihf/c++/4.8/bits/atomic_word.h \
    /usr/include/c++/4.8/bits/locale_classes.h \
    /usr/include/c++/4.8/string \
    /usr/include/c++/4.8/bits/allocator.h \
    /usr/include/arm-linux-gnueabihf/c++/4.8/bits/c++allocator.h \
    /usr/include/c++/4.8/ext/new_allocator.h \
    /usr/include/c++/4.8/bits/ostream_insert.h \
    /usr/include/c++/4.8/bits/cxxabi_forced.h \
    /usr/include/c++/4.8/bits/stl_function.h \
    /usr/include/c++/4.8/backward/binders.h \
    /usr/include/c++/4.8/bits/range_access.h \
    /usr/include/c++/4.8/bits/basic_string.h \
    /usr/include/c++/4.8/bits/basic_string.tcc \
    /usr/include/c++/4.8/bits/locale_classes.tcc \
    /usr/include/c++/4.8/streambuf \
    /usr/include/c++/4.8/bits/streambuf.tcc \
    /usr/include/c++/4.8/bits/basic_ios.h \
    /usr/include/c++/4.8/bits/locale_facets.h \
    /usr/include/c++/4.8/cwctype \
    /usr/include/wctype.h \
    /usr/include/arm-linux-gnueabihf/c++/4.8/bits/ctype_base.h \
    /usr/include/c++/4.8/bits/streambuf_iterator.h \
    /usr/include/arm-linux-gnueabihf/c++/4.8/bits/ctype_inline.h \
    /usr/include/c++/4.8/bits/locale_facets.tcc \
    /usr/include/c++/4.8/bits/basic_ios.tcc \
    /usr/include/c++/4.8/bits/ostream.tcc \
    /usr/include/c++/4.8/istream \
    /usr/include/c++/4.8/bits/istream.tcc \
    /usr/include/c++/4.8/ctime \
    /usr/include/c++/4.8/numeric \
    /usr/include/c++/4.8/bits/stl_numeric.h \
    /usr/include/opencv2/opencv.hpp \
    /usr/include/opencv2/core.hpp \
    /usr/include/opencv2/core/cvdef.h \
    /usr/include/opencv2/core/hal/interface.h \
    /usr/include/c++/4.8/cstddef \
    /usr/lib/gcc/arm-linux-gnueabihf/4.8/include/stdint.h \
    /usr/include/stdint.h \
    /usr/include/opencv2/core/version.hpp \
    /usr/include/opencv2/core/base.hpp \
    /usr/include/c++/4.8/climits \
    /usr/include/c++/4.8/algorithm \
    /usr/include/c++/4.8/utility \
    /usr/include/c++/4.8/bits/stl_relops.h \
    /usr/include/c++/4.8/bits/stl_algo.h \
    /usr/include/c++/4.8/bits/algorithmfwd.h \
    /usr/include/c++/4.8/bits/stl_heap.h \
    /usr/include/c++/4.8/bits/stl_tempbuf.h \
    /usr/include/c++/4.8/bits/stl_construct.h \
    /usr/include/c++/4.8/ext/alloc_traits.h \
    /usr/include/opencv2/core/cvstd.hpp \
    /usr/include/c++/4.8/cstring \
    /usr/include/opencv2/core/ptr.inl.hpp \
    /usr/include/opencv2/core/neon_utils.hpp \
    /usr/include/opencv2/core/traits.hpp \
    /usr/include/opencv2/core/matx.hpp \
    /usr/include/opencv2/core/saturate.hpp \
    /usr/include/opencv2/core/fast_math.hpp \
    /usr/include/opencv2/core/types.hpp \
    /usr/include/c++/4.8/cfloat \
    /usr/lib/gcc/arm-linux-gnueabihf/4.8/include/float.h \
    /usr/include/c++/4.8/vector \
    /usr/include/c++/4.8/bits/stl_uninitialized.h \
    /usr/include/c++/4.8/bits/stl_vector.h \
    /usr/include/c++/4.8/bits/stl_bvector.h \
    /usr/include/c++/4.8/bits/vector.tcc \
    /usr/include/opencv2/core/mat.hpp \
    /usr/include/opencv2/core/bufferpool.hpp \
    /usr/include/opencv2/core/mat.inl.hpp \
    /usr/include/opencv2/core/persistence.hpp \
    /usr/include/opencv2/core/operations.hpp \
    /usr/include/c++/4.8/cstdio \
    /usr/include/opencv2/core/cvstd.inl.hpp \
    /usr/include/c++/4.8/complex \
    /usr/include/c++/4.8/sstream \
    /usr/include/c++/4.8/bits/sstream.tcc \
    /usr/include/opencv2/core/utility.hpp \
    /usr/include/opencv2/core/core_c.h \
    /usr/include/opencv2/core/types_c.h \
    /usr/include/opencv2/core/optim.hpp \
    /usr/include/opencv2/imgproc.hpp \
    /usr/include/opencv2/imgproc/imgproc_c.h \
    /usr/include/opencv2/imgproc/types_c.h \
    /usr/include/opencv2/photo.hpp \
    /usr/include/opencv2/photo/photo_c.h \
    /usr/include/opencv2/video.hpp \
    /usr/include/opencv2/video/tracking.hpp \
    /usr/include/opencv2/video/background_segm.hpp \
    /usr/include/opencv2/video/tracking_c.h \
    /usr/include/opencv2/features2d.hpp \
    /usr/include/opencv2/flann/miniflann.hpp \
    /usr/include/opencv2/flann/defines.h \
    /usr/include/opencv2/flann/config.h \
    /usr/include/opencv2/objdetect.hpp \
    /usr/include/opencv2/objdetect/detection_based_tracker.hpp \
    /usr/include/opencv2/objdetect/objdetect_c.h \
    /usr/include/c++/4.8/deque \
    /usr/include/c++/4.8/bits/stl_deque.h \
    /usr/include/c++/4.8/bits/deque.tcc \
    /usr/include/opencv2/calib3d.hpp \
    /usr/include/opencv2/core/affine.hpp \
    /usr/include/opencv2/calib3d/calib3d_c.h \
    /usr/include/opencv2/imgcodecs.hpp \
    /usr/include/opencv2/videoio.hpp \
    /usr/include/opencv2/highgui.hpp \
    /usr/include/opencv2/highgui/highgui_c.h \
    /usr/include/opencv2/imgcodecs/imgcodecs_c.h \
    /usr/include/opencv2/videoio/videoio_c.h \
    /usr/include/opencv2/ml.hpp \
    /usr/include/c++/4.8/map \
    /usr/include/c++/4.8/bits/stl_tree.h \
    /usr/include/c++/4.8/bits/stl_map.h \
    /usr/include/c++/4.8/bits/stl_multimap.h \
    /usr/include/opencv2/cudaimgproc.hpp \
    /usr/include/opencv2/core/cuda.hpp \
    /usr/include/opencv2/core/cuda_types.hpp \
    /usr/include/opencv2/core/cuda.inl.hpp \
    /usr/include/opencv2/cudacodec.hpp
