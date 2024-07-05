#define NN_IMPLEMENTATION
#include "nn-h.h"
/*
typedef struct
{
    Mat a0;
    Mat w1,b1,a1;
    Mat w2,b2,a2;
}Xor;
Xor xor_alloc(void)
{
    Xor m;

    m.a0 = mat_alloc(1,2);
    

    m.w1 = mat_alloc(2,2);
    m.b1 = mat_alloc(1,2);
    m.a1= mat_alloc(1,2);

    m.w2 = mat_alloc(2,1);
    m.b2 = mat_alloc(1,1);
    m.a2= mat_alloc(1,1);
    return m;
}
float forward(Xor m)
{

    mat_dot(m.a1,m.a0,m.w1);
    mat_sum(m.a1,m.b1);
    mat_sig(m.a1);

    mat_dot(m.a2,m.a1,m.w2);
    mat_sum(m.a2,m.b2);
    mat_sig(m.a2);

    return *(m.a2.es);
}
float cost(Xor m,Mat ti,Mat to)
{
    assert(ti.rows==to.rows);
    assert(to.cols==m.a2.cols);
    size_t n= ti.rows;


    float c=0.f;
    for(size_t i=0;i<n;i++)
    {
        Mat x = mat_row(ti,i);
        Mat y = mat_row(to,i);

        mat_copy(m.a0,x);
        forward(m);

        size_t q=to.cols;
        for(size_t j=0;j<q;j++)
        {
            float d = MAT_AT(m.a2,0,j)-MAT_AT(y,0,j);
            c+=d*d;

        }

    }
    return c/n;
}

void finite_diff(Xor m,Xor g,float eps,Mat ti,Mat to)
{
    float saved;
    float c = cost(m,ti,to);
    
    for(size_t i=0;i<m.w1.rows;i++)
    {
        for(size_t j=0;j<m.w1.cols;j++)
        {
            saved = MAT_AT(m.w1,i,j);
            MAT_AT(m.w1,i,j) += eps;
            MAT_AT(g.w1,i,j)=(cost(m,ti,to)-c)/eps;
            MAT_AT(m.w1,i,j)=saved;
        }
    }
    for(size_t i=0;i<m.b1.rows;i++)
    {
        for(size_t j=0;j<m.b1.cols;j++)
        {
            saved = MAT_AT(m.b1,i,j);
            MAT_AT(m.b1,i,j) += eps;
            MAT_AT(g.b1,i,j)=(cost(m,ti,to)-c)/eps;
            MAT_AT(m.b1,i,j)=saved;
        }
    }
    for(size_t i=0;i<m.w2.rows;i++)
    {
        for(size_t j=0;j<m.w2.cols;j++)
        {
            saved = MAT_AT(m.w2,i,j);
            MAT_AT(m.w2,i,j) += eps;
            MAT_AT(g.w2,i,j)=(cost(m,ti,to)-c)/eps;
            MAT_AT(m.w2,i,j)=saved;
        }
    }
    for(size_t i=0;i<m.b2.rows;i++)
    {
        for(size_t j=0;j<m.b2.cols;j++)
        {
            saved = MAT_AT(m.b2,i,j);
            MAT_AT(m.b2,i,j) += eps;
            MAT_AT(g.b2,i,j)=(cost(m,ti,to)-c)/eps;
            MAT_AT(m.b2,i,j)=saved;
        }
    }
}
void learn(Xor m,Xor g,float rate)
{
    for(size_t i=0;i<m.w1.rows;i++)
    {
        for(size_t j=0;j<m.w1.cols;j++)
        { 
            MAT_AT(m.w1,i,j) -= MAT_AT(g.w1,i,j)*rate;
        }
    }
    for(size_t i=0;i<m.w2.rows;i++)
    {
        for(size_t j=0;j<m.w2.cols;j++)
        { 
            MAT_AT(m.w2,i,j) -= MAT_AT(g.w2,i,j)*rate;
        }
    }
    for(size_t i=0;i<m.b1.rows;i++)
    {
        for(size_t j=0;j<m.b1.cols;j++)
        { 
            MAT_AT(m.b1,i,j) -= MAT_AT(g.b1,i,j)*rate;
        }
    }
    for(size_t i=0;i<m.b2.rows;i++)
    {
        for(size_t j=0;j<m.b2.cols;j++)
        { 
            MAT_AT(m.b2,i,j) -= MAT_AT(g.b2,i,j)*rate;
        }
    }
}
*/
float td[] ={
    0,0,0,0,
    0,1,0,1,
    1,0,0,1,
    1,1,1,0
};

int main()
{
    system("cls");
    
    float eps=2e-1;
    float rate=10e-1;

    size_t stride = 4 ;
    size_t n = sizeof(td)/sizeof(td[0])/stride;
    Mat ti=(Mat){.rows=n,.cols = 2 ,.stride = stride,.es=td};
    Mat to=(Mat){.rows=n,.cols = 2 ,.stride =stride, .es = td + 2};

    MAT_PRINT(ti);
    MAT_PRINT(to);



    size_t arch[]={2,2,2};
    NN nn = nn_alloc(arch,ARRAY_LEN(arch));
    NN g = nn_alloc(arch,ARRAY_LEN(arch));
    
    
    size_t t=time(0);
    
    #if 1//for multiple start
    NN_RAND(nn);
    float pc= nn_cost(nn,ti,to);
    for(size_t i=0;i<100000;i++)
    {
        size_t new=(time(0)+i*time(0))*i*i*3.1554;
        srand(new);
        NN_RAND(nn);
        if(nn_cost(nn,ti,to)<pc){
            t=new;
            srand(t);
            NN_RAND(nn);
            nn_forward(nn);
            printf("prev: %f , now: %f\n",pc,nn_cost(nn,ti,to));
            pc=nn_cost(nn,ti,to);
        }
    }
    #endif
    srand(t);
    NN_RAND(nn);
    nn_forward(nn);

    
    printf("cost: %f\n",nn_cost(nn,ti,to));


    for(size_t i = 0; i <10000000 ; i++)
    {
        nn_learn(nn,g,rate);
        nn_backprop(nn,g,ti,to);
        //nn_finite_diff(nn,g,1e-2,ti,to);
        
    }
    //NN_PRINT(nn);
    printf("cost: %f\n",nn_cost(nn,ti,to));
    for(size_t i = 0; i<2;i++)
    {
        for(size_t j = 0; j<2;j++)
        {
            MAT_AT(NN_INPUT(nn),0,0)=i;
            MAT_AT(NN_INPUT(nn),0,1)=j;
            nn_forward(nn);
            printf("%zu + %zu = %f %f\n",i,j,MAT_AT(NN_OUTPUT(nn),0,0),MAT_AT(NN_OUTPUT(nn),0,1));
        }
    }  
    NN_PRINT(nn);
    /*
    float x;
    do{
    printf("\nx = ");
    scanf("%f",&x);
    MAT_AT(NN_INPUT(nn),0,0)=x;
    nn_forward(nn);
    printf("exp(%f) = %f ",x,MAT_AT(NN_OUTPUT(nn),0,0));
    }while(x+1!=0);*/
    return 1;
}