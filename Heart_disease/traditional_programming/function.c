#include "function.h"

int	ft_strlen(char *str)
{
	int i = 0;

	while (str[i] != '\0')
		i++;
	return(i);
}

int main(int argc, char **argv)
{
	if (argc != 2)
	{
		printf(RED "Error\n" RESET);
		exit(1);
	}
	char *str = argv[1];
	int my_length = ft_strlen(str);

	printf(YELLOW "string = %s\n", str);
	printf(GREEN "string_len = %d\n", my_length);

	return(0);
}
