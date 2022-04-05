/* 
 * This is implementation of a linear time algorithm for the projection 
 * onto the intersection of the L1-ball and L-infinity ball
 * (the latter is also known as a box constraint):
 *    C = { x \in R^n  :  ||x||_1 \le \alpha,  ||x||_{\infty} \le 1 }
 *
 * More details about this algorithm can be found in the following file:
 *    https://github.com/anastasia-podosinnikova/projection-L1-Linf/files/13646/script.pdf
 *
 * This file is a part of the following repository:
 *    https://github.com/anastasia-podosinnikova/projection-L1-Linf
 *
 * This code is distributed under the Appach 2.0 licence.
 * If you use this algorithm/code for your research, please, cite:
 *    A. Podosinnikova. Robust Principal Component Analysis as a Nonlinear
 *    Eigenproblem. Master's Thesis, Saarland University, Department of 
 *    Mathematics and Computer Science, 2013.
 *
 * In case you have some questions or problems with this code, contact:
 *    firstname.lastname@inria.fr / ens.fr
 */

/* Copyright 2015, Anastasia Podosinnikova */

#include <math.h>
#include <mex.h>
#include <matrix.h>

/* a - a pointer to the addresses of x; temp - some of the addresses */
static void swap(double** a, int i, int j)
{ double* temp=a[i]; a[i]=a[j]; a[j]=temp; }

/* *a[i] - value of x with address a[i] */
static void compare(double** a, int i, int j)
{ if (*a[i] < *a[j]) {swap(a,i,j);} }

static int medianOfFive(double** a, int l)
{
  compare(a,l,l+1);                                        /* 1st comp */
  compare(a,l+2,l+3);                                      /* 2nd comp */
  if (*a[l]<*a[l+2])                                       /* 3rd comp */
  {swap(a,l,l+2); swap(a,l+1,l+3);}
  compare(a,l+1,l+4);                                      /* 4th comp */
  if (*a[l+1]>*a[l+2]){                                    /* 5th comp */                                        
    if (*a[l+2]>*a[l+4]) {return l+2;} else {return l+4;}  /* 6th comp */
  } else{
    if (*a[l+1]>*a[l+3]) {return l+1;} else {return l+3;}  /* 6th comp */
  }
}

static void insertionSort(double** a, int l, int r)
{
  int i,j; double pivot; double* pivotAdd; /* as a is an array of addresses */
  for (i=l+1;i<=r;i++){
    pivot=*a[i]; pivotAdd=a[i]; j=i-1;
    while (j>=l){
      if (*a[j]>pivot) {a[j+1]=a[j]; a[j]=pivotAdd;}
      j--;
    }
  }
  return;
}

static int medianOfMedians(double** a, int l, int r)
{
  int i; int medianInd; 
  int M=(r-l+1)/5; /* M = # 5-groups */
  /* Note that M might be increased by one below */
  if (M<=1){ /* recursion exit criterion */
    insertionSort(a,l,r);
    medianInd=l+(r-l)/2;
  } else{
    int medOfFive;
    for (i=0;i<M;i++){
      medOfFive=medianOfFive(a,l+i*5);
      swap(a,l+i,medOfFive); /* move in the beginning */
    }
    
    /* the last group (size < 5) */
    int sizeLast=(r-l)+1-M*5;
    if (sizeLast>2){
      insertionSort(a,l+M*5,r);
      swap(a,l+M,l+M*5+sizeLast/2);
      M++;
    }
    medianInd = medianOfMedians(a,l,l+M-1);
  }
  return medianInd;
}

static void partition(double** a, int l, int m, int r, int* res)
{
  double median = *a[m]; swap(a,m,r); /* move median in the end */
  int i=l-1; /* below median */
  int j=l; /* unknown or above median */
  int k=r; /* equal to median */
  while (j<k){
    if (*a[j]<median) {i++; swap(a,i,j); j++;}
    else if (*a[j]==median) {k--; swap(a,j,k);}
    else {j++;}
  }
  /* placing all A(=p) between A(<p) and A(>p) */
  int t; for(t=0;t<=r-k;t++) {swap(a,i+t+1,k+t);}
  res[0]=i+1; res[1]=i+r-k+1; /* r-k is the # medians (in the end) */
}

/* changes order of elements in x and in w (weights)! */
static void median(double **xadd, int left, int right, int* res)
{
  int i;
  int N = right - left + 1;
  int M = left + N/2; if (N % 2 == 0) {M --;} /* lower median */
  /* int M = left + N/2; */ /* upper median */
  int l = left; int r = right; 
  int* momInds = new int[2]; int momInd, momIndL, momIndR;
  
  while (1>0){
    momInd = medianOfMedians(xadd,l,r);
    partition(xadd,l,momInd,r,momInds);
    momIndL = momInds[0]; momIndR = momInds[1];
    if (momIndL>M) {r=momIndL-1;}
    if (momIndR<M) {l=momIndR+1;}
    if ((momIndL<=M)&&(momIndR>=M)) {break;}
  }
  delete[] momInds; res[0]=momIndL; res[1]=momIndR;
}

static void findX(double* x, double* v, int N, double theta, int* signv)
{
  int i; for(i=0;i<N;i++){
    x[i]=v[i]-theta;
    if (x[i]<0) {x[i]=0;}
    if (x[i]>1) {x[i]=1;}
    x[i]=x[i]*signv[i]; /* go back from the non-negative orthant to the whole space */
  }
}

static int sign(double x){
  return (( x > 0 ) - ( x < 0 ));
}

/* z=alpha; v=b (in the derivation of the algorithm) */
/* project onto intersection of (alpha)z-L1-ball and 1-LInf-ball */
static void project(double* x, double* v, double z, int N)
{
  int i;
  // first, check in v \in C
  double norm1=0; double normInf=0; double absvi;
  for (i=0;i<N;i++){
    absvi=fabs(v[i]); norm1+=absvi;
    if (absvi>normInf) {normInf=absvi;}
  }
  if ((norm1<=z)&(normInf<=1)){
    for (i=0;i<N;i++){x[i]=v[i];}
    return;
  }
  
  /* now, if v \not\in C */
  /* transform to the non-negative orthant */
  int* signv = new int[N]; double sumx=0;
  for (i=0;i<N;i++){ 
    signv[i]=sign(v[i]); v[i]=v[i]*signv[i]; 
    if (v[i]>1){x[i]=1;} else {x[i]=v[i];} /* this x for case theta=0 */
    sumx+=x[i];
  }
  /* check if theta=0 */
  if (sumx<=z){
    for (i=0;i<N;i++){x[i]=x[i]*signv[i];}
    delete[] signv;
    return;
  }
  
  /* case theta > 0 */
  
  /* construct vvb array & find its min(thetaL) and max(thetaR) */
  /* only non-negative values included (dual feasibility: theta>=0) */
  int counter=0;
  for (i=0;i<N;i++){if(v[i]>=1){counter++;}}
  int K=N+counter; /* the length of vvb array */
  double* vvb = new double[K];
  double thetaL=DBL_MAX; double thetaR=-DBL_MAX; /* requires <float.h> */
  for (i=0;i<N;i++){
    vvb[i]=v[i]; 
    if(v[i]>thetaR){thetaR=v[i];} /* max(thetaR) is among v[i]'s */
  } 
  counter=N; for (i=0;i<N;i++){
    if(v[i]>=1){vvb[counter]=v[i]-1;counter++;} 
    if(v[i]-1<thetaL){thetaL=v[i]-1;} /* min(thetaL) is among (v[i]-1)'s */
  }
  
  /* vvbadd is an array of adresses of vvb[left]...vvb[right] */
  int left=0; int right=K-1;
  double** vvbadd = new double*[K];
  for (i=0;i<K;i++) {vvbadd[i]=&vvb[i];}
  
  /* vadd - addresses of v[leftv]...v[rightv] */
  int leftv=0; int rightv=N-1;
  double** vadd = new double*[K];
  for (i=0;i<N;i++) {vadd[i]=&v[i];}
  
  double S=0; for(i=0;i<N;i++) {S=S+v[i];}
  double SL=0; double SR=0; int nL=0; int nR=0;
  double SLTemp, SRTemp; int nLTemp, nRTemp;
  int cardL, cardM, cardU; 
  double sumvM, thetaP, zP;
  int *medInds = new int[2];
  int medLInd, medRInd;
  double theta = -1;
  int iter = 0;
  int il,ic,ir;
  while (1>0){
    median(vvbadd,left,right,medInds);
    medLInd=medInds[0]; medRInd=medInds[1];
    thetaP = *vvbadd[medLInd];
    
    SLTemp=0; SRTemp=0; nLTemp=0; nRTemp=0;
    il=leftv-1; ic=leftv; ir=rightv+1;
    while (ic<ir){
      if (*vadd[ic]<thetaP){il++; swap(vadd,ic,il); SLTemp=SLTemp+*vadd[il]; nLTemp++; ic++;}
      else if (*vadd[ic]>thetaP+1){ir--; swap(vadd,ic,ir); SRTemp=SRTemp+*vadd[ir]; nRTemp++;}
      else {ic++;}
    }
    
    cardU = nR+nRTemp; cardL = nL+nLTemp; cardM = N-cardU-cardL;
    sumvM = S-SLTemp-SRTemp;
    zP = cardU+sumvM-thetaP*cardM;
    
    if (zP>z)
    {SL=SL+SLTemp; S=S-SLTemp; nL=nL+nLTemp; thetaL=thetaP; left=medRInd; leftv=il+1;}
    if (zP < z)
    {SR=SR+SRTemp; S=S-SRTemp; nR=nR+nRTemp; thetaR=thetaP; right=medLInd; rightv=ir-1;}
    
    if (zP == z){theta = thetaP; break;}
    if (right-left<=1) {break;}
    if ((right==medRInd)&(left==medLInd)){break;} /* median(thetaP) is not unique */
    iter++;if (iter == 100){printf("PROJ MAXITER: %d (error)\n",iter); break;}
  }
  
  if (theta==-1) /* theta != to one of the vvb[i] */
  {  
    double zL,zR;
    double thetaL=*vvbadd[left]; double thetaR=*vvbadd[right];
    
    double SLTempL=0; double SRTempL=0; int nLTempL=0; int nRTempL=0;
    il=leftv-1; ic=leftv; ir=rightv+1;
    while (ic<ir){
      if (*vadd[ic]<=thetaL){il++; swap(vadd,ic,il); SLTempL=SLTempL+*vadd[il]; nLTempL++; ic++;}
      else if (*vadd[ic]>=thetaL+1){ir--; swap(vadd,ic,ir); SRTempL=SRTempL+*vadd[ir]; nRTempL++;}
      else {ic++;}
    }
    cardU = nR+nRTempL; cardL = nL+nLTempL; cardM = N-cardU-cardL;
    sumvM = S-SLTempL-SRTempL;
    zL = cardU+sumvM-thetaL*cardM;
    
    double SLTempR=0; double SRTempR=0; int nLTempR=0; int nRTempR=0;
    il=leftv-1; ic=leftv; ir=rightv+1;
    while (ic<ir){
      if (*vadd[ic]<=thetaR){il++; swap(vadd,ic,il); SLTempR=SLTempR+*vadd[il]; nLTempR++; ic++;}
      else if (*vadd[ic]>=thetaR+1){ir--; swap(vadd,ic,ir); SRTempR=SRTempR+*vadd[ir]; nRTempR++;}
      else {ic++;}
    }
    cardU = nR+nRTempR; cardL = nL+nLTempR; cardM = N-cardU-cardL;
    sumvM = S-SLTempR-SRTempR;
    zR = cardU+sumvM-thetaR*cardM;
    
    if (z>zL){nR+=nRTempL+nRTempR; SR+=SRTempL+SRTempR; S-=(SRTempL+SRTempR);} 
    if (z<zR){nL+=nLTempL+nLTempR; SL+=SLTempL+SLTempR; S-=(SLTempL+SLTempR);}
    if ((z>zR)&(z<zL)){nL+=nLTempL; SL+=SLTempL; nR+=nRTempR; SR+=SRTempR; S-=(SLTempL+SRTempR);}
    
    theta=(nR+S-z)/(N-nR-nL); /* S is sumvM by construction */
  }
  
  findX(x,v,N,theta,signv);
  delete[] vvbadd; delete[] vadd; delete[] medInds; delete[] vvb; delete[] signv;
}

/*
 * input: b - vectorToProject, alpha - radiusOfL1Ball
 * output: P_C(b) - projectedVector
 *
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray*prhs[])
{
  if (nrhs!=2)            {mexErrMsgTxt("2 input arguments are reguired");}
  if (nlhs>1)             {mexErrMsgTxt("wrong number of outputs");}
  if ((int)mxGetN(prhs[0])>1) {mexErrMsgTxt("The vectorToProject should be a column vector.");}
  
  int N = (int)mxGetM(prhs[0]);
  double *vectorToProject  = mxGetPr(prhs[0]);
  double radiusOfL1Ball = mxGetScalar(prhs[1]);
  if (radiusOfL1Ball<=0) {mexErrMsgTxt("The radius L1-ball should be positive.");}
  
  plhs[0] = mxCreateDoubleMatrix(N,1,mxREAL);
  double* projectedVector = mxGetPr(plhs[0]);
  
  project(projectedVector, vectorToProject, radiusOfL1Ball, N);
}
