NAME := $(notdir $(lastword $(abspath .)))
BIN  := bin/$(NAME)
SRC  := src
HDR  := ../../src

HDRS_C   := $(wildcard $(SRC)/*.h) $(wildcard $(HDR)/*.h)
HDRS_CPP := $(HDRS_C) $(wildcard $(SRC)/*.hpp) $(wildcard $(HDR)/*.hpp)
SRCS     := $(wildcard $(SRC)/*.S) $(wildcard $(SRC)/*.c) $(wildcard $(SRC)/*.cpp)
OBJS     := $(SRCS:%=%.o)

AS       := $(AS)
ASFLAGS  :=
CC       := clang
CCFLAGS  := -Wall -Wfatal-errors -Ofast -std=c11 -I$(HDR)
CXX      := clang++
CXXFLAGS := -Wall -Wfatal-errors -Ofast -std=c++14 -I$(HDR)
LD       := clang++
LDFLAGS  :=

DIR_PLOT := plot

EXT_OMEGA := omega

DIMENSIONS = $(if $(wildcard $(BIN)),$(shell $(BIN) dimensions),unknown-dimensions)

OUTPUT_ROBUST     = $(DIR_PLOT)/$(DIMENSIONS)_makefile.$(EXT_OMEGA)
OUTPUT_TRANSFERT := $(DIR_PLOT)/transfert.dat

GNUPLOT_ROBUST    := $(DIR_PLOT)/omega.gp
GNUPLOT_TRANSFERT := $(DIR_PLOT)/transfert.gp
PLOT_ROBUST        = $(OUTPUT_ROBUST).svg
PLOT_TRANSFERT    := $(OUTPUT_TRANSFERT).svg

SIGMOID_K := 1

.PHONY: build clean robust robust-plot plot

build: $(BIN)
clean:
	$(RM) $(OBJS) $(BIN)
robust: $(OUTPUT_ROBUST)
robust-plot: $(PLOT_ROBUST)
	@eog $(PLOT_ROBUST)&
plot: $(PLOT_TRANSFERT)
	@eog $(PLOT_TRANSFERT)&

%.S.o: %.S Makefile
	$(AS) $(ASFLAGS) -o $@ $<
%.c.o: %.c $(HDRS_C) Makefile
	$(CC) $(CCFLAGS) -c -o $@ $<
%.cpp.o: %.cpp $(HDRS_CPP) Makefile
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BIN): $(OBJS) Makefile
	$(LD) $(LDFLAGS) -o $@ $(OBJS)

$(OUTPUT_ROBUST): $(BIN) Makefile
	$(BIN) robust $(SIGMOID_K) > $(OUTPUT_ROBUST)
$(OUTPUT_TRANSFERT): $(BIN) Makefile
	$(BIN) plot $(SIGMOID_K) > $(OUTPUT_TRANSFERT)

$(PLOT_ROBUST): $(OUTPUT_ROBUST) Makefile
	gnuplot -e "filename='$(OUTPUT_ROBUST)'" $(GNUPLOT_ROBUST)
$(PLOT_TRANSFERT): $(OUTPUT_TRANSFERT) Makefile
	gnuplot -e "filename='$(OUTPUT_TRANSFERT)'" $(GNUPLOT_TRANSFERT)
