# minimalist makefile
.SUFFIXES:
#
.SUFFIXES: .cpp .o .c .h
ifeq ($(DEBUG),1)
CFLAGS = -fPIC  -std=c99 -ggdb  -march=native -Wall -Wextra -pedantic -Wshadow  -mavx512f -mavx512dq -fsanitize=undefined  -fno-omit-frame-pointer -fsanitize=address
else
CFLAGS = -fPIC -std=c99 -O3 -Wall -Wextra -pedantic -Wshadow -mavx512f -mavx512dq
endif # debug

HEADERS=include/simdpcg32.h  include/pcg32.h


all: fillarray

%: ./benchmark/%.c $(HEADERS)
	$(CC) $(CFLAGS) -o $@ $<  -Iinclude

%: ./benchmark/%.cpp $(HEADERS)
	$(CC) $(CFLAGS) -std=c++17 -o $@ $< -Iinclude

clean:
	rm -f  fillarray fillarr
