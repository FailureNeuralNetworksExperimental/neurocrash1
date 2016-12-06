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

DIR_NET    := net
DIR_PLOT   := plot
DIR_DATA   := data
DIR_IMAGES := img

EXT_NET     := net
EXT_EPSILON := epsilon

DATA_TRAIN_IMAGES := $(DIR_DATA)/train-images
DATA_TRAIN_LABELS := $(DIR_DATA)/train-labels
DATA_TEST_IMAGES  := $(DIR_DATA)/test-images
DATA_TEST_LABELS  := $(DIR_DATA)/test-labels
DATA_TRAIN := $(DATA_TRAIN_IMAGES) $(DATA_TRAIN_LABELS)
DATA_TEST  := $(DATA_TEST_IMAGES) $(DATA_TEST_LABELS)

DIMENSIONS = $(if $(wildcard $(BIN)),$(shell $(BIN) dimensions),unknown-dimensions)

OUTPUT_TRAIN      = $(DIR_NET)/$(DIMENSIONS)_batch-$(BATCH_SIZE)_makefile.$(EXT_NET)
OUTPUT_ROBUST     = $(DIR_PLOT)/$(DIMENSIONS)_batch-$(BATCH_SIZE)_makefile.$(EXT_EPSILON)
OUTPUT_TRANSFERT := $(DIR_PLOT)/transfert.dat

GNUPLOT_ROBUST    := $(DIR_PLOT)/epsilon.gp
GNUPLOT_TRANSFERT := $(DIR_PLOT)/transfert.gp
PLOT_ROBUST        = $(OUTPUT_ROBUST).svg
PLOT_TRANSFERT    := $(OUTPUT_TRANSFERT).svg

SIGMOID_K  := 1
BATCH_SIZE := 10
RAND_LIMIT := 0.01

.PHONY: build clean download-database train train-rand test test-images test-images-clean robust robust-plot transfert-plot dimensions-get

build: $(BIN)
clean:
	$(RM) $(OBJS) $(BIN)
download-database: $(DATA_TRAIN) $(DATA_TEST)
train: $(OUTPUT_TRAIN)
train-rand: $(BIN)
	$(BIN) rand $(RAND_LIMIT) > $(OUTPUT_TRAIN)
test: $(BIN) $(OUTPUT_TRAIN) $(DATA_TEST)
	$(BIN) test $(DATA_TEST) $(SIGMOID_K) < $(OUTPUT_TRAIN)
test-images: $(BIN) $(OUTPUT_TRAIN) $(DATA_TEST)
	$(BIN) test $(DATA_TEST) $(SIGMOID_K) "$(DIR_IMAGES)" < $(OUTPUT_TRAIN)
test-images-clean:
	$(RM) $(DIR_IMAGES)/*.pgm
robust: $(OUTPUT_ROBUST)
robust-plot: $(PLOT_ROBUST)
	@eog $(PLOT_ROBUST)&
transfert-plot: $(PLOT_TRANSFERT)
	@eog $(PLOT_TRANSFERT)&
dimensions-get: $(BIN)
	@echo Supported network dimensions: $(shell $(BIN) dimensions ", ")

%.S.o: %.S Makefile
	$(AS) $(ASFLAGS) -o $@ $<
%.c.o: %.c $(HDRS_C) Makefile
	$(CC) $(CCFLAGS) -c -o $@ $<
%.cpp.o: %.cpp $(HDRS_CPP) Makefile
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BIN): $(OBJS) Makefile
	$(LD) $(LDFLAGS) -o $@ $(OBJS)

$(DATA_TRAIN_IMAGES):
	wget -q -O - "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz" | gzip -d -c - > $(DATA_TRAIN_IMAGES)
$(DATA_TRAIN_LABELS):
	wget -q -O - "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz" | gzip -d -c - > $(DATA_TRAIN_LABELS)
$(DATA_TEST_IMAGES):
	wget -q -O - "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz" | gzip -d -c - > $(DATA_TEST_IMAGES)
$(DATA_TEST_LABELS):
	wget -q -O - "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz" | gzip -d -c - > $(DATA_TEST_LABELS)

$(OUTPUT_TRAIN): $(BIN) $(DATA_TRAIN) Makefile
	$(BIN) train $(DATA_TRAIN) $(SIGMOID_K) $(BATCH_SIZE) > $(OUTPUT_TRAIN)
$(OUTPUT_ROBUST): $(BIN) $(OUTPUT_TRAIN) $(DATA_TEST) Makefile
	$(BIN) robust $(DATA_TEST) $(SIGMOID_K) < $(OUTPUT_TRAIN) > $(OUTPUT_ROBUST)
$(OUTPUT_TRANSFERT): $(BIN) Makefile
	$(BIN) plot $(SIGMOID_K) > $(OUTPUT_TRANSFERT)

$(PLOT_ROBUST): $(OUTPUT_ROBUST) Makefile
	gnuplot -e "filename='$(OUTPUT_ROBUST)'" $(GNUPLOT_ROBUST)
$(PLOT_TRANSFERT): $(OUTPUT_TRANSFERT) Makefile
	gnuplot -e "filename='$(OUTPUT_TRANSFERT)'" $(GNUPLOT_TRANSFERT)
