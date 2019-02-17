# minimalist makefile
.SUFFIXES:
#
.SUFFIXES: .cpp .o .c .h
ifeq ($(DEBUG),1)
CFLAGS = -fPIC  -ggdb  -march=native -Wall -Wextra -pedantic -Wshadow -mavx2 -mavx512f -mavx512dq -mavx512vl -fsanitize=undefined  -fno-omit-frame-pointer -fsanitize=address
else
CFLAGS = -fPIC -O3 -Wall -Wextra -pedantic -Wshadow -mavx2 -mavx512f -mavx512dq -mavx512vl
endif # debug

HEADERS=include/simdpcg32.h  include/pcg32.h


all: fillarray

%: ./benchmark/%.c $(HEADERS)
	$(CC) $(CFLAGS) -std=c99 -o $@ $<  -Iinclude

%cpp: ./benchmark/%.cpp $(HEADERS)
	$(CC) $(CFLAGS) -std=c++17 -S -fverbose-asm -o $@ $< -Iinclude && \
    echo "This file isn't executable." && \
    echo "Remove -S -verbose-asm to generate an executable for testing " && \
    echo "if you're on a computer supporting AVX512{dq,vl,f}" \

clean:
	rm -f  fillarray fillarr
