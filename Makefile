COMPILER = mpicc
OPT_FLAGS = -std=c11 -O3
DEBUG_FLAGS = -Wall -Wextra -Wno-unused-parameter
HDF5_FLAGS = -I$(shell pwd)/hdf5/include -L$(shell pwd)/hdf5/lib -l:libhdf5.a -lz -lsz
LIBS_FLAGS = -lm
SRC = $(wildcard main.c src/*.c)

run: clean main.out
	mpirun -np 6 ./main.out

main.out: 
	$(COMPILER) $(SRC) -o $@ $(OPT_FLAGS) $(DEBUG_FLAGS) $(HDF5_FLAGS) $(LIBS_FLAGS)

clean:
	rm -f main.out

cleandata:
	rm -f *.h5

install_hdf5:
	mkdir -p hdf5
	tar -xvzf archives/hdf5* -C hdf5 --strip-components=2
	cd hdf5;\
	CC=mpicc ./configure --enable-parallel --enable-shared;\
	make;\
	make install;\
	cd ../;\
	mv hdf5/hdf5/include hdf5/;\
	mv hdf5/hdf5/lib hdf5/;\