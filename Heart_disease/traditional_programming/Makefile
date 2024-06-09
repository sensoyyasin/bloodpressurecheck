CC	=	gcc
CFLAGS = -Wall -Wextra -Werror

SOURCES = $(wildcard *.c)
HEADERS = $(wildcard *.h)

OBJECTS = $(SOURCES:.c=.o)
EXECUTABLE = function

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(CFLAGS) $(OBJECTS) -o $@

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

clean_func:
	rm -f $(EXECUTABLE).o

clean: clean_func
	rm -rf $(EXECUTABLE) $(OBJECTS)
