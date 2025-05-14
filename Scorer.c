#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LINE		100
#define MAX_UNT_LIST	2000

int		nunt;
char	unTlist[MAX_UNT_LIST][MAX_LINE];

void Read_UNT_List(char *fn)
{
	FILE	*unF;
	char	line[MAX_LINE];

	unF = fopen(fn,"r");
	nunt = 0;
	while (fgets(line,MAX_LINE,unF) != NULL) {
		strcpy(unTlist[nunt],strtok(line," \t\n"));
		nunt++;
	}
	fclose(unF);
}


int	IsUT(char *t)
{
	int low = 0;
	int high = nunt;
	int p = (high+low)/2;
	int	pr = 0;
	while ((low < high) && (pr != p)) {
		if (strcmp(unTlist[p],t) > 0)
			high = p;
		else 
			if (strcmp(unTlist[p],t) < 0)  
				low = p;
			else
				return (1);
		pr = p;
		p = (high+low)/2;
	}
	return (0);
}


/*******************************************************************/
int main(int argc, char **argv)
{
	char	line1[MAX_LINE],line2[MAX_LINE];
	char	tok1[100],tok2[100],tag1[20],tag2[20];
	int		ok,nok,ut_ok,ut_nok,j;
	FILE	*f1,*f2;
	float	acc,errR,ut_acc,ut_errR;

	if (argc < 3) {
		fprintf(stderr,"usage: Compare <tagged file> <reference file> [<Unknown token list>]\n");
		exit(1);
	}

	// READ UNKNOWN TOKEN LIST
	if (argc < 4)
		nunt = 0;
	else
		Read_UNT_List(argv[3]);

	f1 = fopen(argv[1],"r");
	f2 = fopen(argv[2],"r");
	j = ok = nok = ut_ok = ut_nok = 0;
	while (fgets(line1,MAX_LINE,f1) != NULL) {
		j++;
		fgets(line2,MAX_LINE,f2);

		strcpy(tok1,strtok(line1," \t\n"));
		strcpy(tag1,strtok(NULL," \t\n"));

		strcpy(tok2,strtok(line2," \t\n"));
		strcpy(tag2,strtok(NULL," \t\n"));

		if (strcmp(tok1,tok2) != 0) {
			fprintf(stderr,"ERROR: The two files are not aligned at position %d!\n",j);
			exit(1);
		}
		else 
			if (strcmp(tag1,tag2) == 0) {
				ok++;
				printf("OK-");
				if (IsUT(tok1)) 
					ut_ok++;
			}
			else {
				nok++;
				printf("***************NOK-");
				if (IsUT(tok1)) 
					ut_nok++;
			}
		printf("%s (%s) - %s (%s)\n",line1,tag1,line2,tag2);
	}
	fclose(f2);
	fclose(f1);

	acc  = (float)ok*100/(nok+ok);
	errR = (float)nok*100/(nok+ok);
	ut_acc  = (float)ut_ok*100/(ut_nok+ut_ok);
	ut_errR = (float)ut_nok*100/(ut_nok+ut_ok);
	fprintf(stderr,"GLOBAL DATA: %d differences on %d tokens\n",nok,nok+ok);
	fprintf(stderr,"Accuracy = %5.2f	Error Rate = %5.2f\n",acc,errR);
	if (ut_nok+ut_ok != 0) {
		fprintf(stderr,"UNKNOWN TOKENS: %d differences on %d tokens\n",ut_nok,ut_nok+ut_ok);
		fprintf(stderr,"UTAccuracy = %5.2f	UTError Rate = %5.2f\n",ut_acc,ut_errR);
	}
}
