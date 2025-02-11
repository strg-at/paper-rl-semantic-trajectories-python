all: cffi glove
cffi:
	cd glove_ffi && python cooccur_extension_build.py && cd ..
glove:
	cd submodules/GloVe && make && cd .. && cd ..
clean:
	cd glove_ffi && rm _crec.* cooccur.o && cd ..
	cd submodules/Glove && make clean
