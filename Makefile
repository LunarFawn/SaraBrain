CXX = g++
CXXFLAGS = -O3 -fPIC -std=c++17
LDFLAGS = -shared

TARGET = src/sara_brain/core/sara_engine.so
SRC = src/sara_brain/core/engine.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)
