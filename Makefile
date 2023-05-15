CC = g++
CFLAGS = -lm -fopenmp
TARGET = all04

all: $(TARGET)

$(TARGET): $(TARGET).cpp
	$(CC) $(TARGET).cpp -o $(TARGET) $(CFLAGS)

clean:
	rm -f $(TARGET)
