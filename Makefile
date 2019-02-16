# minimalist makefile
.SUFFIXES:
#
.SUFFIXES: .cpp .o .c .h
ifeq ($(DEBUG),1)
CFLAGS = -fPIC  -std=c99 -ggdb  -march=native -Wall -Wextra -pedantic -Wshadow -mavx2 -mavx512f -mavx512dq -mavx512vl -fsanitize=undefined  -fno-omit-frame-pointer -fsanitize=address
else
CFLAGS = -fPIC -std=c99 -O3 -Wall -Wextra -pedantic -Wshadow -mavx2 -mavx512f -mavx512dq -mavx512vl
endif # debug

HEADERS=include/simdpcg32.h  include/pcg32.h


all: fillarray

%: ./benchmark/%.c $(HEADERS)
	$(CC) $(CFLAGS) -o $@ $<  -Iinclude

%cpp: ./benchmark/%.cpp $(HEADERS)
	$(CC) $(CFLAGS) -std=c++17 -S -fverbose-asm -o $@ $< -Iinclude

clean:
	rm -f  fillarray fillarr
