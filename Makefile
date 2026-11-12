# compile all codes with "make", pythia8-config and root-config have to be in PATH

CXX      = g++
CXXFLAGS = -O2 -std=c++17
PYTHIA   = $(shell pythia8-config --cxxflags --libs)
ROOTLIBS = $(shell root-config --cflags --libs)

SRC = $(wildcard */*/dis_*.cc)
BIN = $(SRC:.cc=)

all: $(BIN)

%: %.cc
	$(CXX) $(CXXFLAGS) $< -o $@ $(PYTHIA) $(ROOTLIBS)

clean:
	rm -f $(BIN)
