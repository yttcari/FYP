// This is the source file where the methods computing the primary spectra are implemented.
// Last modification: 27 May 2026
// Authors: A. Arbey (alexandre.arbey@ens-lyon.fr), M. Calza, L. Malacher, D. Pedrotti, Y.F. Perez-Gonzalez
// Authors: A. Arbey & J. Auffinger


#include "include.h"

void read_gamma_tables(double ***gammas,double *gamma_param,double *gamma_x,struct param *parameters){
	// This function reads the greybody factors tables stored in the folder '/gamma_tables'.
	// It fills the tabulated BH spins, energies and factors in the arrays
	// gamma_param[], gamma_x[] and gammas[][][] respectively.
	
	char **gamma_names = (char **)malloc(5*sizeof(char *)); // contains the names of the greybody tables files
	for(int i = 0;i<parameters->nb_gamma_spins;i++){
		gamma_names[i] = (char *)malloc(64*sizeof(char));
	}
	switch(parameters->metric){
		case 0:{ // Kerr BHs
			sprintf(gamma_names[0],"./src/tables/gamma_tables/spin_0.txt");
			sprintf(gamma_names[1],"./src/tables/gamma_tables/spin_1.txt");
			sprintf(gamma_names[2],"./src/tables/gamma_tables/spin_2.txt");
			sprintf(gamma_names[3],"./src/tables/gamma_tables/spin_0.5.txt");
			sprintf(gamma_names[4],"./src/tables/gamma_tables/spin_1.5.txt");
			break;
		}
		case 1:{ // polymerized BHs
			if(parameters->a0_LQG == 0.){ // a0=0
				sprintf(gamma_names[0],"./src/tables/gamma_tables/LQG/spin_0.txt");
				sprintf(gamma_names[1],"./src/tables/gamma_tables/LQG/spin_1.txt");
				sprintf(gamma_names[2],"./src/tables/gamma_tables/LQG/spin_2.txt");
				sprintf(gamma_names[3],"./src/tables/gamma_tables/LQG/spin_0.5.txt");
			}
			else{ // a0=0.11
				sprintf(gamma_names[0],"./src/tables/gamma_tables/LQG/spin_0_a0.txt");
				sprintf(gamma_names[1],"./src/tables/gamma_tables/LQG/spin_1_a0.txt");
				sprintf(gamma_names[2],"./src/tables/gamma_tables/LQG/spin_2_a0.txt");
				sprintf(gamma_names[3],"./src/tables/gamma_tables/LQG/spin_0.5_a0.txt");
			}
			break;
		}
		case 2:{ // charged BHs
			sprintf(gamma_names[0],"./src/tables/gamma_tables/charged/spin_0.txt");
			sprintf(gamma_names[1],"./src/tables/gamma_tables/charged/spin_1.txt");
			sprintf(gamma_names[2],"./src/tables/gamma_tables/charged/spin_2.txt");
			sprintf(gamma_names[3],"./src/tables/gamma_tables/charged/spin_0.5.txt");
			break;
		}
		case 3:{ // higher-dimensional BHs
			sprintf(gamma_names[0],"./src/tables/gamma_tables/higher/spin_0.txt");
			sprintf(gamma_names[1],"./src/tables/gamma_tables/higher/spin_1.txt");
			sprintf(gamma_names[2],"./src/tables/gamma_tables/higher/spin_2.txt");
			sprintf(gamma_names[3],"./src/tables/gamma_tables/higher/spin_0.5.txt");
			break;
		}
		case 4:{ // Bardeen BHs
			sprintf(gamma_names[0],"./src/tables/gamma_tables/bardeen/spin_0.txt");
			sprintf(gamma_names[1],"./src/tables/gamma_tables/bardeen/spin_1.txt");
			sprintf(gamma_names[2],"./src/tables/gamma_tables/bardeen/spin_2.txt");
			sprintf(gamma_names[3],"./src/tables/gamma_tables/bardeen/spin_0.5.txt");
			break;
		}
		case 5:{ // Hayward BHs
			sprintf(gamma_names[0],"./src/tables/gamma_tables/hayward/spin_0.txt");
			sprintf(gamma_names[1],"./src/tables/gamma_tables/hayward/spin_1.txt");
			sprintf(gamma_names[2],"./src/tables/gamma_tables/hayward/spin_2.txt");
			sprintf(gamma_names[3],"./src/tables/gamma_tables/hayward/spin_0.5.txt");
			break;
		}
		case 6:{ // Simpson-Visser BHs
			sprintf(gamma_names[0],"./src/tables/gamma_tables/simpsonvisser/spin_0.txt");
			sprintf(gamma_names[1],"./src/tables/gamma_tables/simpsonvisser/spin_1.txt");
			sprintf(gamma_names[2],"./src/tables/gamma_tables/simpsonvisser/spin_2.txt");
			sprintf(gamma_names[3],"./src/tables/gamma_tables/simpsonvisser/spin_0.5.txt");
			break;
		}
		case 7:{ // Peltola-Kunstatter BHs
			sprintf(gamma_names[0],"./src/tables/gamma_tables/peltolakunstatter/spin_0.txt");
			sprintf(gamma_names[1],"./src/tables/gamma_tables/peltolakunstatter/spin_1.txt");
			sprintf(gamma_names[2],"./src/tables/gamma_tables/peltolakunstatter/spin_2.txt");
			sprintf(gamma_names[3],"./src/tables/gamma_tables/peltolakunstatter/spin_0.5.txt");
			break;
		}
		case 8:{ // D'Ambrosio-Rovelli BHs
			sprintf(gamma_names[0],"./src/tables/gamma_tables/dambrosiorovelli/spin_0.txt");
			sprintf(gamma_names[1],"./src/tables/gamma_tables/dambrosiorovelli/spin_1.txt");
			sprintf(gamma_names[2],"./src/tables/gamma_tables/dambrosiorovelli/spin_2.txt");
			sprintf(gamma_names[3],"./src/tables/gamma_tables/dambrosiorovelli/spin_0.5.txt");
			break;
		}
		case 9:{ // BCL BHs
			sprintf(gamma_names[0],"./src/tables/gamma_tables/BCL/spin_0.txt");
			sprintf(gamma_names[1],"./src/tables/gamma_tables/BCL/spin_1.txt");
			sprintf(gamma_names[2],"./src/tables/gamma_tables/BCL/spin_2.txt");
			sprintf(gamma_names[3],"./src/tables/gamma_tables/BCL/spin_0.5.txt");
			break;
		}
		default:{
			printf("\n\t [read_gamma_tables] : ERROR WRONG BH METRIC !\n");
			fflush(stdout);
			exit(0);
			break;
		}
	}
	
	char dummy[32];
	FILE *table;
	if(parameters->full_output){
		printf("\n\n\t reading:\t");
		fflush(stdout);
	}
	for(int i = 0;i<parameters->nb_gamma_spins;i++){
		table = fopen(gamma_names[i],"r");
		if(!table){
			printf("\n\t [read_gamma_tables] : ERROR COULD NOT FIND TABLE '%s' !\n",gamma_names[i]);
			fflush(stdout);
			exit(0);
		}
		rewind(table);
		fscanf(table,"%s",dummy);
		for(int j = 0;j<parameters->nb_gamma_x;j++){
			fscanf(table,"%lf",&(gamma_x[j]));
		}
		for(int j = 0;j<parameters->nb_gamma_param;j++){
			fscanf(table,"%lf",&(gamma_param[j]));
			for(int k = 0;k<parameters->nb_gamma_x;k++){
				fscanf(table,"%lf",&(gammas[i][j][k]));
			}
		}
		fclose(table);
		if(parameters->full_output){ // prints the state of reading
			if(i == 0){
				printf("%s\n",gamma_names[i]);
			}
			else{
				printf("\t\t\t%s\n",gamma_names[i]);
			}
			fflush(stdout);
		}
	}
	if(parameters->full_output){
		printf("\n");
		fflush(stdout);
	}
	free2D_char(gamma_names,parameters->nb_gamma_spins);
	return;
}

void read_asymp_fits(double ***fits,struct param *parameters){
	// This function reads the tables fitting asymptotic Kerr emissivities
	// in the folder "/gamma_tables". It fills the array fits[][][].
	
	char **table_names;
	table_names = (char **)malloc(parameters->nb_gamma_spins*sizeof(char *)); // contains the names of the fits tables files
	for(int i = 0;i<parameters->nb_gamma_spins;i++){
		table_names[i] = (char *)malloc(64*sizeof(char));
	}
	switch(parameters->metric){
		case 0:{ // Kerr BHs
			sprintf(table_names[0],"./src/tables/gamma_tables/spin_0_fits.txt");
			sprintf(table_names[1],"./src/tables/gamma_tables/spin_1_fits.txt");
			sprintf(table_names[2],"./src/tables/gamma_tables/spin_2_fits.txt");
			sprintf(table_names[3],"./src/tables/gamma_tables/spin_0.5_fits.txt");
			sprintf(table_names[4],"./src/tables/gamma_tables/spin_1.5_fits.txt");
			break;
		}
		case 1:{ // polymerized BHs
			if(parameters->a0_LQG == 0.){ // a0=0
				sprintf(table_names[0],"./src/tables/gamma_tables/LQG/spin_0_fits.txt");
				sprintf(table_names[1],"./src/tables/gamma_tables/LQG/spin_1_fits.txt");
				sprintf(table_names[2],"./src/tables/gamma_tables/LQG/spin_2_fits.txt");
				sprintf(table_names[3],"./src/tables/gamma_tables/LQG/spin_0.5_fits.txt");
			}
			else{ // a0=0.11
				sprintf(table_names[0],"./src/tables/gamma_tables/LQG/spin_0_fits_a0.txt");
				sprintf(table_names[1],"./src/tables/gamma_tables/LQG/spin_1_fits_a0.txt");
				sprintf(table_names[2],"./src/tables/gamma_tables/LQG/spin_2_fits_a0.txt");
				sprintf(table_names[3],"./src/tables/gamma_tables/LQG/spin_0.5_fits_a0.txt");
			}
			break;
		}
		case 2:{ // charged BHs
			sprintf(table_names[0],"./src/tables/gamma_tables/charged/spin_0_fits.txt");
			sprintf(table_names[1],"./src/tables/gamma_tables/charged/spin_1_fits.txt");
			sprintf(table_names[2],"./src/tables/gamma_tables/charged/spin_2_fits.txt");
			sprintf(table_names[3],"./src/tables/gamma_tables/charged/spin_0.5_fits.txt");
			break;
		}
		case 3:{ // higher dimensional BHs
			sprintf(table_names[0],"./src/tables/gamma_tables/higher/spin_0_fits.txt");
			sprintf(table_names[1],"./src/tables/gamma_tables/higher/spin_1_fits.txt");
			sprintf(table_names[2],"./src/tables/gamma_tables/higher/spin_2_fits.txt");
			sprintf(table_names[3],"./src/tables/gamma_tables/higher/spin_0.5_fits.txt");
			break;
		}
		case 4:{ // Bardeen BHs
			sprintf(table_names[0],"./src/tables/gamma_tables/bardeen/spin_0_fits.txt");
			sprintf(table_names[1],"./src/tables/gamma_tables/bardeen/spin_1_fits.txt");
			sprintf(table_names[2],"./src/tables/gamma_tables/bardeen/spin_2_fits.txt");
			sprintf(table_names[3],"./src/tables/gamma_tables/bardeen/spin_0.5_fits.txt");
			break;
		}		
		case 5:{ // Hayward BHs
			sprintf(table_names[0],"./src/tables/gamma_tables/hayward/spin_0_fits.txt");
			sprintf(table_names[1],"./src/tables/gamma_tables/hayward/spin_1_fits.txt");
			sprintf(table_names[2],"./src/tables/gamma_tables/hayward/spin_2_fits.txt");
			sprintf(table_names[3],"./src/tables/gamma_tables/hayward/spin_0.5_fits.txt");
			break;
		}
		case 6:{ // Simpson-Visser BHs
			sprintf(table_names[0],"./src/tables/gamma_tables/simpsonvisser/spin_0_fits.txt");
			sprintf(table_names[1],"./src/tables/gamma_tables/simpsonvisser/spin_1_fits.txt");
			sprintf(table_names[2],"./src/tables/gamma_tables/simpsonvisser/spin_2_fits.txt");
			sprintf(table_names[3],"./src/tables/gamma_tables/simpsonvisser/spin_0.5_fits.txt");
			break;
		}		
		case 7:{ // Peltola-Kunstatter BHs
			sprintf(table_names[0],"./src/tables/gamma_tables/peltolakunstatter/spin_0_fits.txt");
			sprintf(table_names[1],"./src/tables/gamma_tables/peltolakunstatter/spin_1_fits.txt");
			sprintf(table_names[2],"./src/tables/gamma_tables/peltolakunstatter/spin_2_fits.txt");
			sprintf(table_names[3],"./src/tables/gamma_tables/peltolakunstatter/spin_0.5_fits.txt");
			break;
		}			
		case 8:{ // D'Ambrosio-Rovelli BHs
			sprintf(table_names[0],"./src/tables/gamma_tables/dambrosiorovelli/spin_0_fits.txt");
			sprintf(table_names[1],"./src/tables/gamma_tables/dambrosiorovelli/spin_1_fits.txt");
			sprintf(table_names[2],"./src/tables/gamma_tables/dambrosiorovelli/spin_2_fits.txt");
			sprintf(table_names[3],"./src/tables/gamma_tables/dambrosiorovelli/spin_0.5_fits.txt");
			break;
		}		
		case 9:{ // BCL BHs
			sprintf(table_names[0],"./src/tables/gamma_tables/BCL/spin_0_fits.txt");
			sprintf(table_names[1],"./src/tables/gamma_tables/BCL/spin_1_fits.txt");
			sprintf(table_names[2],"./src/tables/gamma_tables/BCL/spin_2_fits.txt");
			sprintf(table_names[3],"./src/tables/gamma_tables/BCL/spin_0.5_fits.txt");
			break;
		}		
		default:{
			printf("\n\t [read_asymp_fits] : ERROR WRONG BH METRIC !\n");
			fflush(stdout);
			exit(0);
			break;
		}
	}
	
	FILE *file;
	char dummy[32];
	if(parameters->full_output){
		printf("\n\n\t reading:\t");
		fflush(stdout);
	}
	for(int i = 0;i<parameters->nb_gamma_spins;i++){
		file = fopen(table_names[i],"r");
		if(file == 0){
			printf("\n\t [read_asymp_fits] : ERROR COULD NOT OPEN FILE '%s' !\n",table_names[i]);
			fflush(stdout);
			exit(0);
		}
		rewind(file);
		for(int j = 0;j<1+parameters->nb_gamma_fits;j++){
			fscanf(file,"%s",dummy);
		}
		for(int j = 0;j<parameters->nb_gamma_param;j++){
			fscanf(file,"%s",dummy);
			for(int k = 0;k<parameters->nb_gamma_fits;k++){
				fscanf(file,"%lf",&(fits[i][j][k]));
			}
		}
		if(parameters->full_output){ // prints the state of reading
			if(i == 0){
				printf("%s\n",table_names[i]);
			}
			else{
				printf("\t\t\t%s\n",table_names[i]);
			}
			fflush(stdout);
		}
		fclose(file);
	}
	if(parameters->full_output){
		printf("\n");
		fflush(stdout);
	}
	free2D_char(table_names,parameters->nb_gamma_spins);
	return;
}

void generate_asymp_fits(double ***gammas,double *gamma_param,double *gamma_x,double ***newfits,struct param *parameters){
		
	double a1,b1,c1,a[5];
	double param;
	double gamma_low1,gamma_low2,gamma_high,gamma[30];
	double x1,x2,xhigh,x[30];
	double T,E1,E2,Ehigh,E[30];
	double M=1.;
	
	int step=10;


	int ix2=1;
	
	x1=gamma_x[0];
	x2=gamma_x[ix2];
		
	if(parameters->metric==0) // Kerr BHs
	{
			for(int k=0;k<10;k++) x[k]=gamma_x[parameters->nb_gamma_x-1-k*step]; // decreasing energies
	}
	else // Spherically symmetric BHs
	{
		xhigh=gamma_x[parameters->nb_gamma_x-1];
	}
	
	for(int i = 0;i<parameters->nb_gamma_spins;i++){
		for(int j = 0;j<parameters->nb_gamma_param;j++){
	
			param=gamma_param[j];
		
			switch(parameters->metric){
				case 0:{ // Kerr BHs
					T = temp_Kerr(M,param);
					break;
				}
				case 1:{ // polymerized BHs
					T = temp_LQG(M,100.*param,parameters->a0_LQG);
					break;
				}
				case 2:{ // charged BHs
					T = temp_charged(M,param);
					break;
				}
				case 3:{ // higher BHs
					T = temp_higher(M,param,parameters->M_star);
					break;
				}
				case 4:{ // Bardeen BHs
					T = temp_bardeen(M,param);
					break;
				}
				case 5:{ // Hayward BHs
					T = temp_hayward(M,param);
					break;
				}
				case 6:{ // Simpson-Visser BHs
					T = temp_simpsonvisser(M,param);
					break;
				}
				case 7:{ // Peltola-Kunstatter BHs
					T = temp_peltolakunstatter(M,param);
					break;
				}
				case 8:{ // D'Ambrosio-Rovelli BHs
					T = temp_dambrosiorovelli(M,param);
					break;
				}
				case 9:{ // BCL BHs
					T = temp_BCL(M,param);
					break;
				}
				default:{
					printf("\n\t [generate_asymp_fits] : ERROR WRONG BH METRIC !\n");
					fflush(stdout);
					exit(0);
					break;
				}
			}
			
			gamma_low1=gammas[i][j][0];
			gamma_low2=gammas[i][j][ix2];
			E1=x1/2./G/M;
			E2=x2/2./G/M;
			
			if(parameters->metric==0) // Kerr BHs
			{
				for(int k=0;k<10;k++)
				{
					gamma[k]=gammas[i][j][parameters->nb_gamma_x-1-k*step];
					E[k]=x[k]/2./G/M;
				}
			}
			else // Spherically symmetric BHs
			{
				gamma_high=gammas[i][j][parameters->nb_gamma_x-1];
				Ehigh=xhigh/2./G/M;
			}
								
			if(i <= 2){ // bosons
				
				// Low energy
				if(parameters->metric==0)
				{
					a1 = (log10(gamma_low2) - log10(gamma_low1))/(log10(x2)-log10(x1));
					b1 = log10(gamma_low1) - a1*log10(x1);
				}
				else	
				{
					a1 = (log10(gamma_low2*exp_adapt(E2/T)) - log10(gamma_low1*exp_adapt(E1/T)))/(log10(x2)-log10(x1));
					b1 = log10(gamma_low1/2.*exp_adapt(E1/T)) - a1*log10(x1);
				}		
				
				// High energy
				if(parameters->metric==0) // Kerr BHs
				{					
					double a0[20],a1[20],a2[20],a3[20],a4[20],test[20],test2[20];;
					for(int k=0;k<20;k++) a0[k]=test[k]=test2[k]=0.;
						
					for(step=1;step<20;step++)
					{
						for(int k=0;k<10;k++)
						{
							x[k]=gamma_x[parameters->nb_gamma_x-1-k*step]; // decreasing energies
							gamma[k]=gammas[i][j][parameters->nb_gamma_x-1-k*step];
							E[k]=x[k]/2./G/M;
						}

						double y[10];
						for(int k=0;k<10;k++) y[k]=log10(gamma[k]);
						
						double h=log10(x[0])-log10(x[1]); // step size between successive log10(x)
						
						double yp[9]; // derivative wrt x
						for(int k=0;k<9;k++) yp[k]=(y[k+1]-y[k])/(x[k+1]-x[k]);
						
						double ypp[8]; // second derivative wrt x
						for(int k=0;k<8;k++) ypp[k]=(yp[k+1]-yp[k])/(x[k+1]-x[k]);
						
						a[0]=((y[2]-y[0])*(ypp[2]-ypp[1])-(y[2]-y[1])*(ypp[2]-ypp[0]))/((x[2]-x[0])*(ypp[2]-ypp[1])-(x[2]-x[1])*(ypp[2]-ypp[0]));
						
						a[1]=	((y[2]*ypp[1]-y[1]*ypp[2])*(x[2]*ypp[0]-x[0]*ypp[2])
								-(y[2]*ypp[0]-y[0]*ypp[2])*(x[2]*ypp[1]-x[1]*ypp[2]))
								/
								((ypp[1]-ypp[2])*(x[2]*ypp[0]-x[0]*ypp[2])
								-(ypp[0]-ypp[2])*(x[2]*ypp[1]-x[1]*ypp[2]));
						
						if(((a[1]-y[0]+a[0]*x[0])/ypp[0]>0.)&&a[0]<0.)
						{
						
							a[4]=1./sqrt((a[1]-y[0]+a[0]*x[0])/ypp[0]);
																		
							double z[5],zp[5];
							for(int k=0;k<5;k++) z[k]=y[k]-a[0]*x[k]-a[1];
											
							a[2]=(z[1]*sin(a[4]*x[0])-z[0]*sin(a[4]*x[1]))/(cos(a[4]*x[1])*sin(a[4]*x[0])-cos(a[4]*x[0])*sin(a[4]*x[1]));
							a[3]=(-z[1]*cos(a[4]*x[0])+z[0]*cos(a[4]*x[1]))/(cos(a[4]*x[1])*sin(a[4]*x[0])-cos(a[4]*x[0])*sin(a[4]*x[1]));

							a0[step]=a[0];
							a1[step]=a[1];
							a2[step]=a[2];
							a3[step]=a[3];
							a4[step]=a[4];
							
							for(int k=0;k<10;k++) test[step]+=pow(a[0]*gamma_x[parameters->nb_gamma_x-1-k]+a[1]+a[2]*cos(a[4]*gamma_x[parameters->nb_gamma_x-1-k])+a[3]*sin(a[4]*gamma_x[parameters->nb_gamma_x-1-k])-log10(gammas[i][j][parameters->nb_gamma_x-1-k]),2.);
								
							test2[step]=fabs(1.-a0[step]/yp[0]);
						}
						
					}
					
					for(int k=0;k<5;k++) a[k]=0.;
					
					
					double testmin=1000.;
					int stepmin=0;
					
					for(int k=0;k<10;k++) if(a0[k]!=0.&&test[k]<testmin&&test2[k]<0.1){testmin=test[k];stepmin=k;};
					if(stepmin==0) for(int k=10;k<20;k++) if(a0[k]!=0.&&test[k]<testmin){testmin=test[k];stepmin=k;};
					
					
					a[0]=a0[stepmin];
					a[1]=a1[stepmin];
					a[2]=a2[stepmin];
					a[3]=a3[stepmin];
					a[4]=a4[stepmin];
					
				}
				else // Spherically symmetric BHs
				{
					if(gamma_high<1.e-300||Ehigh/T>700.) // cases where double precision is not enough
					{
						c1=0.;
					}
					else
					{
						c1=gamma_high/(27./4.*pow(xhigh,2.)/exp_adapt(Ehigh/T));
					}
				}
			}
			else
			{ // fermions
				
				// Low energy
				if(parameters->metric==0) // Kerr BHs
				{
					a1 = (log10(gamma_low2) - log10(gamma_low1))/(log10(x2)-log10(x1));
					b1 = log10(gamma_low1) - a1*log10(x1);					
				}
				else  // Spherically symmetric BHs
				{
					a1 = (log10(gamma_low2*(exp(E2/T) + 1.)) - log10(gamma_low1*(exp(E1/T) + 1.)))/(log10(x2)-log10(x1));
					b1 = log10(gamma_low1/2.*(exp(E1/T) + 1.)) - a1*log10(x1);
				}
				
				// High energy
				if(parameters->metric==0) // Kerr BHs
				{					
					double a0[20],a1[20],a2[20],a3[20],a4[20],test[20],test2[20];;
					for(int k=0;k<20;k++) a0[k]=test[k]=test2[k]=0.;
						
					for(step=1;step<20;step++)
					{
						for(int k=0;k<10;k++)
						{
							x[k]=gamma_x[parameters->nb_gamma_x-1-k*step]; // decreasing energies
							gamma[k]=gammas[i][j][parameters->nb_gamma_x-1-k*step];
							E[k]=x[k]/2./G/M;
						}

						double y[10];
						for(int k=0;k<10;k++) y[k]=log10(gamma[k]);
						
						double h=log10(x[0])-log10(x[1]); // step size between successive log10(x)
						
						double yp[9]; // derivative wrt x
						for(int k=0;k<9;k++) yp[k]=(y[k+1]-y[k])/(x[k+1]-x[k]);
						
						double ypp[8]; // second derivative wrt x
						for(int k=0;k<8;k++) ypp[k]=(yp[k+1]-yp[k])/(x[k+1]-x[k]);
						
						a[0]=((y[2]-y[0])*(ypp[2]-ypp[1])-(y[2]-y[1])*(ypp[2]-ypp[0]))/((x[2]-x[0])*(ypp[2]-ypp[1])-(x[2]-x[1])*(ypp[2]-ypp[0]));
						
						a[1]=	((y[2]*ypp[1]-y[1]*ypp[2])*(x[2]*ypp[0]-x[0]*ypp[2])
								-(y[2]*ypp[0]-y[0]*ypp[2])*(x[2]*ypp[1]-x[1]*ypp[2]))
								/
								((ypp[1]-ypp[2])*(x[2]*ypp[0]-x[0]*ypp[2])
								-(ypp[0]-ypp[2])*(x[2]*ypp[1]-x[1]*ypp[2]));
						
						if(((a[1]-y[0]+a[0]*x[0])/ypp[0]>0.)&&a[0]<0.)
						{
						
							a[4]=1./sqrt((a[1]-y[0]+a[0]*x[0])/ypp[0]);
																		
							double z[5],zp[5];
							for(int k=0;k<5;k++) z[k]=y[k]-a[0]*x[k]-a[1];
											
							a[2]=(z[1]*sin(a[4]*x[0])-z[0]*sin(a[4]*x[1]))/(cos(a[4]*x[1])*sin(a[4]*x[0])-cos(a[4]*x[0])*sin(a[4]*x[1]));
							a[3]=(-z[1]*cos(a[4]*x[0])+z[0]*cos(a[4]*x[1]))/(cos(a[4]*x[1])*sin(a[4]*x[0])-cos(a[4]*x[0])*sin(a[4]*x[1]));

							a0[step]=a[0];
							a1[step]=a[1];
							a2[step]=a[2];
							a3[step]=a[3];
							a4[step]=a[4];
							
							for(int k=0;k<10;k++) test[step]+=pow(a[0]*gamma_x[parameters->nb_gamma_x-1-k]+a[1]+a[2]*cos(a[4]*gamma_x[parameters->nb_gamma_x-1-k])+a[3]*sin(a[4]*gamma_x[parameters->nb_gamma_x-1-k])-log10(gammas[i][j][parameters->nb_gamma_x-1-k]),2.);
							
							test2[step]=fabs(1.-a0[step]/yp[0]);
						}
						
					}
					
					for(int k=0;k<5;k++) a[k]=0.;
					
					
					double testmin=1000.;
					int stepmin=0;
					
					for(int k=0;k<10;k++) if(a0[k]!=0.&&test[k]<testmin&&test2[k]<0.1){testmin=test[k];stepmin=k;};
					if(stepmin==0) for(int k=10;k<20;k++) if(a0[k]!=0.&&test[k]<testmin){testmin=test[k];stepmin=k;};
					
					
					a[0]=a0[stepmin];
					a[1]=a1[stepmin];
					a[2]=a2[stepmin];
					a[3]=a3[stepmin];
					a[4]=a4[stepmin];
				}
				else  // Spherically symmetric BHs
				{
					if(gamma_high<1.e-300||Ehigh/T>700.) // cases where double precision is not enough
					{	
						c1=0.;
					}
					else
					{
						c1=gamma_high/(27./4.*pow(xhigh,2.)/(exp(Ehigh/T) + 1.));
					}
				}
			}	
			
			if(parameters->metric==0) // Kerr BHs -> 2 low energy params + 5 high energy params
			{
				newfits[i][j][0]=gamma_param[j];
				newfits[i][j][1]=a1;
				newfits[i][j][2]=b1;
				for(int k=0;k<5;k++) newfits[i][j][k+3]=a[k];
			}	
			else // Spherically symmetric BHs -> 2 low energy params + 1 high energy params
			{
				newfits[i][j][0]=gamma_param[j];
				newfits[i][j][1]=a1;
				newfits[i][j][2]=b1;
				newfits[i][j][3]=c1;
			}		
		}
	}
	
	return;
}

void write_asymp_fits(double ***newfits,struct param *parameters){
	
	char **table_names;
	table_names = (char **)malloc(parameters->nb_gamma_spins*sizeof(char *)); // contains the names of the fits tables files
	for(int i = 0;i<parameters->nb_gamma_spins;i++){
		table_names[i] = (char *)malloc(64*sizeof(char));
	}
	
	switch(parameters->metric){
		case 0:{ // Kerr BHs
			sprintf(table_names[0],"./src/tables/gamma_tables/spin_0_fits_gen.txt");
			sprintf(table_names[1],"./src/tables/gamma_tables/spin_1_fits_gen.txt");
			sprintf(table_names[2],"./src/tables/gamma_tables/spin_2_fits_gen.txt");
			sprintf(table_names[3],"./src/tables/gamma_tables/spin_0.5_fits_gen.txt");
			sprintf(table_names[4],"./src/tables/gamma_tables/spin_1.5_fits_gen.txt");
			break;
		}
		case 1:{ // polymerized BHs
			if(parameters->a0_LQG == 0.){ // a0=0
				sprintf(table_names[0],"./src/tables/gamma_tables/LQG/spin_0_fits_gen.txt");
				sprintf(table_names[1],"./src/tables/gamma_tables/LQG/spin_1_fits_gen.txt");
				sprintf(table_names[2],"./src/tables/gamma_tables/LQG/spin_2_fits_gen.txt");
				sprintf(table_names[3],"./src/tables/gamma_tables/LQG/spin_0.5_fits_gen.txt");
			}
			else{ // a0=0.11
				sprintf(table_names[0],"./src/tables/gamma_tables/LQG/spin_0_fits_a0_gen.txt");
				sprintf(table_names[1],"./src/tables/gamma_tables/LQG/spin_1_fits_a0_gen.txt");
				sprintf(table_names[2],"./src/tables/gamma_tables/LQG/spin_2_fits_a0_gen.txt");
				sprintf(table_names[3],"./src/tables/gamma_tables/LQG/spin_0.5_fits_a0_gen.txt");
			}
			break;
		}
		case 2:{ // Charged BHs
			sprintf(table_names[0],"./src/tables/gamma_tables/charged/spin_0_fits_gen.txt");
			sprintf(table_names[1],"./src/tables/gamma_tables/charged/spin_1_fits_gen.txt");
			sprintf(table_names[2],"./src/tables/gamma_tables/charged/spin_2_fits_gen.txt");
			sprintf(table_names[3],"./src/tables/gamma_tables/charged/spin_0.5_fits_gen.txt");
			break;
		}		
		case 3:{ // Higher dimensional BHs
			sprintf(table_names[0],"./src/tables/gamma_tables/higher/spin_0_fits_gen.txt");
			sprintf(table_names[1],"./src/tables/gamma_tables/higher/spin_1_fits_gen.txt");
			sprintf(table_names[2],"./src/tables/gamma_tables/higher/spin_2_fits_gen.txt");
			sprintf(table_names[3],"./src/tables/gamma_tables/higher/spin_0.5_fits_gen.txt");
			break;
		}	
		case 4:{ // Bardeen BHs
			sprintf(table_names[0],"./src/tables/gamma_tables/bardeen/spin_0_fits_gen.txt");
			sprintf(table_names[1],"./src/tables/gamma_tables/bardeen/spin_1_fits_gen.txt");
			sprintf(table_names[2],"./src/tables/gamma_tables/bardeen/spin_2_fits_gen.txt");
			sprintf(table_names[3],"./src/tables/gamma_tables/bardeen/spin_0.5_fits_gen.txt");
			break;
		}
		case 5:{ // Hayward BHs
			sprintf(table_names[0],"./src/tables/gamma_tables/hayward/spin_0_fits_gen.txt");
			sprintf(table_names[1],"./src/tables/gamma_tables/hayward/spin_1_fits_gen.txt");
			sprintf(table_names[2],"./src/tables/gamma_tables/hayward/spin_2_fits_gen.txt");
			sprintf(table_names[3],"./src/tables/gamma_tables/hayward/spin_0.5_fits_gen.txt");
			break;
		}
		case 6:{ // Simpson-Visser BHs
			sprintf(table_names[0],"./src/tables/gamma_tables/simpsonvisser/spin_0_fits_gen.txt");
			sprintf(table_names[1],"./src/tables/gamma_tables/simpsonvisserd/spin_1_fits_gen.txt");
			sprintf(table_names[2],"./src/tables/gamma_tables/simpsonvisser/spin_2_fits_gen.txt");
			sprintf(table_names[3],"./src/tables/gamma_tables/simpsonvisser/spin_0.5_fits_gen.txt");
			break;
		}				
		case 7:{ // Peltola-Kunstatter BHs
			sprintf(table_names[0],"./src/tables/gamma_tables/peltolakunstatter/spin_0_fits_gen.txt");
			sprintf(table_names[1],"./src/tables/gamma_tables/peltolakunstatter/spin_1_fits_gen.txt");
			sprintf(table_names[2],"./src/tables/gamma_tables/peltolakunstatter/spin_2_fits_gen.txt");
			sprintf(table_names[3],"./src/tables/gamma_tables/peltolakunstatter/spin_0.5_fits_gen.txt");
			break;
		}			
		case 8:{ // D'Ambrosio-Rovelli BHs
			sprintf(table_names[0],"./src/tables/gamma_tables/dambrosiorovelli/spin_0_fits_gen.txt");
			sprintf(table_names[1],"./src/tables/gamma_tables/dambrosiorovelli/spin_1_fits_gen.txt");
			sprintf(table_names[2],"./src/tables/gamma_tables/dambrosiorovelli/spin_2_fits_gen.txt");
			sprintf(table_names[3],"./src/tables/gamma_tables/dambrosiorovelli/spin_0.5_fits_gen.txt");
			break;
		}		
		case 9:{ // BCL BHs
			sprintf(table_names[0],"./src/tables/gamma_tables/BCL/spin_0_fits_gen.txt");
			sprintf(table_names[1],"./src/tables/gamma_tables/BCL/spin_1_fits_gen.txt");
			sprintf(table_names[2],"./src/tables/gamma_tables/BCL/spin_2_fits_gen.txt");
			sprintf(table_names[3],"./src/tables/gamma_tables/BCL/spin_0.5_fits_gen.txt");
			break;
		}		
		default:{
			printf("\n\t [write_asymp_fits] : ERROR WRONG BH METRIC !\n");
			fflush(stdout);
			exit(0);
			break;
		}
	}
	
	FILE *file;
	char dummy[32];
	if(parameters->full_output){
		printf("\n\n\t writing:\t");
		fflush(stdout);
	}
	for(int i = 0;i<parameters->nb_gamma_spins;i++){
		file = fopen(table_names[i],"w");
		if(parameters->metric==0) fprintf(file,"l/fitsgen           a1             b1                   a0             a1          a2          a3             a4\n");
		else
		{
			if(parameters->metric==3) fprintf(file,"n/fitsgen           a1             b1          limit\n");
			else fprintf(file,"l/fitsgen           a1             b1          limit\n");
		}
		for(int j = 0;j<parameters->nb_gamma_param;j++){
			if(parameters->metric==0) fprintf(file,"    %.5e    %.5e    %.5e    %.5e    %.5e    %.5e    %.5e    %.5e\n",newfits[i][j][0],newfits[i][j][1],newfits[i][j][2],newfits[i][j][3],newfits[i][j][4],newfits[i][j][5],newfits[i][j][6],newfits[i][j][7]);
			else
			{
				if(parameters->metric==3) fprintf(file,"    %d    %.5e    %.5e    %.5e\n",(int)newfits[i][j][0],newfits[i][j][1],newfits[i][j][2],newfits[i][j][3]);
				else fprintf(file,"    %.5e    %.5e    %.5e    %.5e\n",newfits[i][j][0],newfits[i][j][1],newfits[i][j][2],newfits[i][j][3]);
			}
		}
		if(parameters->full_output){ // prints the state of writing
			if(i == 0){
				printf("%s\n",table_names[i]);
			}
			else{
				printf("\t\t\t%s\n",table_names[i]);
			}
			fflush(stdout);
		}
		fclose(file);
	}
	if(parameters->full_output){
		printf("\n");
		fflush(stdout);
	}
	free2D_char(table_names,parameters->nb_gamma_spins);
	
	return;
}

double dNdtdE(double E,double M,double param,int particle_index,double ***gammas,double *gamma_param,double *gamma_x,double ***fits,double *dof,double *spins,double *masses_primary,int counter_param,int counter_x,struct param *parameters){
	// This function computes the emission rate d²N/dtdE for the elementary particle of type
	// particle_index. It uses the greybody factors tables or the asymptotical fits.
	
	if(M == 0. || (parameters->BH_remnant && M == parameters->M_remnant)){ // the BH has evaporated or is a stable remnant
		return 0.;
	}
	double T;
	double x = 2.*E*M*G;
	switch(parameters->metric){
		case 0:{ // Kerr BHs
			T = temp_Kerr(M,param);
			break;
		}
		case 1:{ // polymerized BHs
			T = temp_LQG(M,100.*param,parameters->a0_LQG);
			break;
		}
		case 2:{ // charged BHs
			T = temp_charged(M,param);
			break;
		}
		case 3:{ // higher BHs
			T = temp_higher(M,param,parameters->M_star);
			break;
		}
		case 4:{ // Bardeen BHs
			T = temp_bardeen(M,param);
			break;
		}
		case 5:{ // Hayward BHs
			T = temp_hayward(M,param);
			break;
		}
		case 6:{ // Simpson-Visser BHs
			T = temp_simpsonvisser(M,param);
			break;
		}
		case 7:{ // Peltola-Kunstatter BHs
			T = temp_peltolakunstatter(M,param);
			break;
		}
		case 8:{ // D'Ambrosio-Rovelli BHs
			T = temp_dambrosiorovelli(M,param);
			break;
		}
		case 9:{ // BCL BHs
			T = temp_BCL(M,param);
			break;
		}
		default:{
			printf("\n\t [dNdtdE] : ERROR WRONG BH METRIC !\n");
			fflush(stdout);
			exit(0);
			break;
		}
	}
	int type; // contains the spin type of the particle
	if(spins[particle_index] == 0.){
		type = 0;
	}
	if(spins[particle_index] == 1.){
		type = 1;
	}
	if(spins[particle_index] == 2.){
		type = 2;
	}
	if(spins[particle_index] == 0.5){
		type = 3;
	}
	if(spins[particle_index] == 1.5){
		type = 4;
	}
	if(E < masses_primary[particle_index]){
		return 0.;
	}
	else if(x > 0.01 && x < 5.){ // tabulated value
		switch(parameters->interpolation_method){
			case 0:{ // linear interpolation
				return dof[particle_index]*(gammas[type][counter_param-1][counter_x-1] + (gammas[type][counter_param][counter_x-1] - gammas[type][counter_param-1][counter_x-1])/(gamma_param[counter_param] - gamma_param[counter_param-1])*(param - gamma_param[counter_param-1]) + (gammas[type][counter_param-1][counter_x] - gammas[type][counter_param-1][counter_x-1])/(gamma_x[counter_x] - gamma_x[counter_x-1])*(2.*E*M*G - gamma_x[counter_x-1]))/(2.*pi);
				break;
			}
			
			case 1:{ // logarithmic interpolation
				if(gammas[type][counter_param-1][counter_x-1] == 0. || gammas[type][counter_param][counter_x-1] == 0. || gammas[type][counter_param-1][counter_x] == 0.){
					return 0.;
				}
				else{
					return dof[particle_index]*(pow(10.,log10(gammas[type][counter_param-1][counter_x-1]) + (log10(gammas[type][counter_param][counter_x-1]) - log10(gammas[type][counter_param-1][counter_x-1]))/(log10(gamma_x[counter_x]) - log10(gamma_x[counter_x-1]))*(log10(2.*E*M*G) - log10(gamma_x[counter_x-1]))) + (gammas[type][counter_param][counter_x-1] - gammas[type][counter_param-1][counter_x-1])/(gamma_param[counter_param] - gamma_param[counter_param-1])*(param - gamma_param[counter_param-1]))/(2.*pi);
				}
				break;
			}
			
			case 2:{ // bilinear interpolation
				// First interpolate in param direction at x=x0
				double f_x0 = gammas[type][counter_param-1][counter_x-1] + 
							  (gammas[type][counter_param][counter_x-1] - 
							   gammas[type][counter_param-1][counter_x-1]) /
							  (gamma_param[counter_param] - gamma_param[counter_param-1]) * 
							  (param - gamma_param[counter_param-1]);
				
				// Then interpolate in param direction at x=x1
				double f_x1 = gammas[type][counter_param-1][counter_x] + 
							  (gammas[type][counter_param][counter_x] - 
							   gammas[type][counter_param-1][counter_x]) /
							  (gamma_param[counter_param] - gamma_param[counter_param-1]) * 
							  (param - gamma_param[counter_param-1]);
				
				// Finally interpolate in x direction
				return dof[particle_index] *
					   (f_x0 + 
						(f_x1 - f_x0) /
						(gamma_x[counter_x] - gamma_x[counter_x-1]) *
						(2.*E*M*G - gamma_x[counter_x-1])) /
					   (2.*pi);
				break;
			}
			
			case 3:{ // semi-logarithmic bilinear interpolation
				if(gammas[type][counter_param-1][counter_x-1] <= 0. || 
				   gammas[type][counter_param][counter_x-1] <= 0. || 
				   gammas[type][counter_param-1][counter_x] <= 0. ||
				   gammas[type][counter_param][counter_x] <= 0.){
					return 0.;
				}
				else{
					// First interpolate in x (log scale) at param=param0
					double log_x_ratio = (log10(2.*E*M*G) - log10(gamma_x[counter_x-1])) /
										 (log10(gamma_x[counter_x]) - log10(gamma_x[counter_x-1]));
					
					double f_param0 = pow(10, log10(gammas[type][counter_param-1][counter_x-1]) +
										  log_x_ratio * (log10(gammas[type][counter_param-1][counter_x]) - 
														 log10(gammas[type][counter_param-1][counter_x-1])));
					
					// Then interpolate in x (log scale) at param=param0
					double f_param1 = pow(10, log10(gammas[type][counter_param][counter_x-1]) +
										  log_x_ratio * (log10(gammas[type][counter_param][counter_x]) - 
														 log10(gammas[type][counter_param][counter_x-1])));
					
					// Finally interpolate in param (linear)
					double param_ratio = (param - gamma_param[counter_param-1]) /
										 (gamma_param[counter_param] - gamma_param[counter_param-1]);
					
					return dof[particle_index] *
						   (f_param0 + param_ratio * (f_param1 - f_param0)) /
						   (2.*pi);
				}
				break;
			}
			
			default:{
				printf("\n\t [dNdtdE] : ERROR WRONG INTERPOLATION METHOD !\n");
				fflush(stdout);
				exit(0);
				break;
			}
		}
	}
	
	
	else if(x <= 0.01)
	{ // low-energy asymptotic value		
		
		double a0=fits[type][counter_param-1][0];
		double b0=fits[type][counter_param-1][1];
		
		double a1=fits[type][counter_param][0];
		double b1=fits[type][counter_param][1];
		
		double fit0,fit1;
		
		
		if(parameters->metric == 0) // Kerr BHs
		{
			fit0=dof[particle_index]*pow(10.,a0*log10(x) + b0)/(2.*pi);
			fit1=dof[particle_index]*pow(10.,a1*log10(x) + b1)/(2.*pi);			
		}
		else // Spherically symmetric BHs
		{
			if(type <= 2){ // bosons
				fit0=dof[particle_index]*pow(10.,a0*log10(x) + b0)/exp_adapt(E/T)/(pi);
				fit1=dof[particle_index]*pow(10.,a1*log10(x) + b1)/exp_adapt(E/T)/(pi);			
			}
			else{ // fermions
				fit0=dof[particle_index]*pow(10.,a0*log10(x) + b0)/(exp(E/T) + 1.)/(pi);
				fit1=dof[particle_index]*pow(10.,a1*log10(x) + b1)/(exp(E/T) + 1.)/(pi);
			}
		}
		
		return fit0 + (fit1 - fit0)/(gamma_param[counter_param] - gamma_param[counter_param-1])*(param - gamma_param[counter_param-1]);
	}
			
		
	else
	{ // high-energy asymptotic value x >= 5.
		
		double fit0,fit1;
		
		if(parameters->metric == 0) // Kerr BHs -> 5 parameters
		{
			double a0=fits[type][counter_param-1][2];
			double b0=fits[type][counter_param-1][3];
			double c0=fits[type][counter_param-1][4];
			double d0=fits[type][counter_param-1][5];
			double e0=fits[type][counter_param-1][6];
			
			double a1=fits[type][counter_param][2];
			double b1=fits[type][counter_param][3];
			double c1=fits[type][counter_param][4];
			double d1=fits[type][counter_param][5];
			double e1=fits[type][counter_param][6];
			
			fit0=dof[particle_index]*pow(10.,a0*(2.*M*E*G) + b0 + c0*cos(e0*2.*E*M*G) + d0*sin(e0*2.*E*M*G))/(2.*pi);
			fit1=dof[particle_index]*pow(10.,a1*(2.*M*E*G) + b1 + c1*cos(e1*2.*E*M*G) + d1*sin(e1*2.*E*M*G))/(2.*pi);
		}
		else // Spherically symmetric BHs
		{
			double c0=fits[type][counter_param-1][2];
			double c1=fits[type][counter_param][2];;
			if(type <= 2){ // bosons
				fit0=dof[particle_index]*(27./4.*pow(x,2.)/exp_adapt(E/T)*c0)/(2.*pi);
				fit1=dof[particle_index]*(27./4.*pow(x,2.)/exp_adapt(E/T)*c1)/(2.*pi);
			}
			else{ // fermions
				fit0=dof[particle_index]*(27./4.*pow(x,2.)/(exp(E/T) + 1.)*c0)/(2.*pi);
				fit1=dof[particle_index]*(27./4.*pow(x,2.)/(exp(E/T) + 1.)*c1)/(2.*pi);
			}
		}
		return fit0 + (fit1 - fit0)/(gamma_param[counter_param] - gamma_param[counter_param-1])*(param - gamma_param[counter_param-1]);
	}
}

void instantaneous_primary_spectrum(double **instantaneous_primary_spectra,double **BH_masses,double **BH_params,double **spec_table,double *energies,
	double ***gammas,double *gamma_param,double *gamma_x,double ***fits,double *dof,double *spins,double *masses_primary,int **counters_param, int ***counters_x,
	int *compute,struct param *parameters){
	// This function computes the instantaneous primary spectra for all particles thanks to the function
	// dNdtdE and the BH spectrum at some timestep.
	
	for(int i = 0;i<parameters->BH_number;i++){
		for(int j = 0;j<parameters->param_number;j++){
			counters_param[i][j] = 0;
			while(counters_param[i][j] < parameters->nb_gamma_param - 1 && BH_params[i][j] >= gamma_param[counters_param[i][j]]){ // finding the nearest parameter
				counters_param[i][j]++;
			}
		}
		for(int j = 0;j<parameters->param_number;j++){
			counters_x[i][j][0] = 0;
			for(int k = 0;k<parameters->E_number;k++){
				if(k>0){
					counters_x[i][j][k] = counters_x[i][j][k-1];
				}
				while(counters_x[i][j][k] < parameters->nb_gamma_x - 1 && energies[k]*2.*BH_masses[i][j]*G >= gamma_x[counters_x[i][j][k]]){ // finding the nearest omega*r_H
					counters_x[i][j][k]++;
				}
			}
		}
	}	
	if(parameters->full_output){
		printf("\n\n\t compute:\t");
		fflush(stdout);
	}
	for(int i = 0;i<parameters->particle_number+parameters->grav+parameters->add_DM;i++){
		for(int j = 0;j<parameters->E_number;j++){
			instantaneous_primary_spectra[i][j] = 0.;
			if(compute[i]){
				for(int k = 0;k<parameters->BH_number;k++){
					for(int l = 0;l<parameters->param_number;l++){
						instantaneous_primary_spectra[i][j] += spec_table[k][l]*dNdtdE(energies[j],BH_masses[k][l],BH_params[k][l],i,gammas,gamma_param,gamma_x,fits,dof,spins,masses_primary,counters_param[k][l],counters_x[k][l][j],parameters);
					}
				}
			}
		}
		if(parameters->full_output){ // prints the current state of computation
			printf("%i/%i ",i+1,parameters->particle_number+parameters->grav+parameters->add_DM);
			fflush(stdout);
		}
	}
	if(parameters->full_output){
		printf("\n\n");
		fflush(stdout);
	}
	return;
}

void write_instantaneous_primary_spectra(double **instantaneous_primary_spectra,double *energies,struct param *parameters){
	// This function writes the instantaneous Hawking primary spectrum of a distribution of BH,
	// contained in the array instantaneous_primary_spectra[][], into the file 'instantaneous_primary_spectra.txt'.
	
	char file_name[500];
	sprintf(file_name,"./results/%s/instantaneous_primary_spectra.txt",parameters->destination_folder);
	FILE *file = fopen(file_name,"w+");
	if(!file){
		printf("\n\t [write_instantaneous_primary_spectra] : ERROR COULD NOT OPEN FILE '%s' !\n",file_name);
		fflush(stdout);
		exit(0);
	}
	rewind(file);
	fprintf(file,"Hawking primary spectra for each particle types.\n");
	if(!parameters->grav){
		if(parameters->add_DM){
			if(parameters->hadronization_choice != 3){
				fprintf(file,"%15s%15s%15s%15s%15s%15s%15s%15s%15s%15s%15s%15s%15s%15s%15s%15s%15s\n","energy/particle","photon","gluons","higgs","W+-","Z0","neutrinos","electron","muon","tau","up","down","charm","strange","top","bottom","DM");
			}
			else{
				fprintf(file,"%15s%15s%15s%15s%15s%15s%15s%15s%15s%15s%15s%15s%15s%15s%15s%15s%15s\n","energy/particle","photon","gluons","higgs","W+-","Z0","neutrinos","electron","muon","tau","up","down","charm","strange","top","bottom","pions");
			}
		}
		else{
			fprintf(file,"%15s%15s%15s%15s%15s%15s%15s%15s%15s%15s%15s%15s%15s%15s%15s%15s\n","energy/particle","photon","gluons","higgs","W+-","Z0","neutrinos","electron","muon","tau","up","down","charm","strange","top","bottom");
		}
	}
	else{
		if(parameters->add_DM){
			fprintf(file,"%15s%15s%15s%15s%15s%15s%15s%15s%15s%15s%15s%15s%15s%15s%15s%15s%15s%15s\n","energy/particle","photon","gluons","higgs","W+-","Z0","neutrinos","electron","muon","tau","up","down","charm","strange","top","bottom","graviton","DM");
		}
		else{
			fprintf(file,"%15s%15s%15s%15s%15s%15s%15s%15s%15s%15s%15s%15s%15s%15s%15s%15s%15s\n","energy/particle","photon","gluons","higgs","W+-","Z0","neutrinos","electron","muon","tau","up","down","charm","strange","top","bottom","graviton");
		}
	}
	for(int j = 0;j<parameters->E_number;j++){
		fprintf(file,"%15.5e",energies[j]);
		for(int i = 0;i<parameters->particle_number+parameters->grav+parameters->add_DM;i++){
			fprintf(file,"%15.5e",instantaneous_primary_spectra[i][j]*rate_conversion); // conversion from GeV to CGS units
		}
		fprintf(file,"\n");
	}
	fclose(file);
	return;
}

