all: cffi glove
cffi:
	cd glove_ffi && python cooccur_extension_build.py && cd ..
glove:
	cd submodules/GloVe && make && cd .. && cd ..
