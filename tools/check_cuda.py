import os, ctypes, pprint, sys

print('Python executable:', sys.executable)
print('CUDA_PATH:', os.environ.get('CUDA_PATH'))
print('\n--- PATH sample (first 20 entries) ---')
for p in os.environ.get('PATH','').split(os.pathsep)[:20]:
    print(' ', p)

print('\n--- Searching PATH for cudart/cudnn DLLs ---')
found = []
for p in os.environ.get('PATH','').split(os.pathsep):
    try:
        for fn in ('cudart64_118.dll','cudart64_110.dll','cudart64_111.dll','cudnn64_8.dll'):
            fp = os.path.join(p, fn)
            if os.path.isfile(fp):
                found.append(fp)
    except Exception:
        pass
if found:
    print('DLLs found:')
    for f in found:
        print(' ', f)
else:
    print('No cudart/cudnn DLLs found on PATH visible to this process.')

print('\n--- Attempting to import TensorFlow ---')
try:
    import tensorflow as tf
    print('tf.__version__:', getattr(tf, '__version__', None))
    try:
        print('tf.test.is_built_with_cuda():', tf.test.is_built_with_cuda())
    except Exception as e:
        print('tf.test.is_built_with_cuda() error:', e)
    try:
        bi = tf.sysconfig.get_build_info()
        print('tf.sysconfig.get_build_info():')
        pprint.pprint(bi)
    except Exception as e:
        print('get_build_info error:', e)
    try:
        print('Physical GPUs:', tf.config.list_physical_devices('GPU'))
    except Exception as e:
        print('tf.config.list_physical_devices error:', e)
except Exception as e:
    print('TensorFlow import ERROR:', repr(e))

print('\n--- Attempt ctypes LoadLibrary on common DLL names ---')
dlls = ['cudart64_118.dll','cudnn64_8.dll']
for d in dlls:
    try:
        ctypes.cdll.LoadLibrary(d)
        print(d, '=> Load OK')
    except Exception as e:
        print(d, '=> Load ERROR:', repr(e))

print('\n--- End diagnostics ---')
