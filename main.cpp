#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>//librería para trabajar con imgs
#include <cstdlib>
#include <fstream>
#include <string.h>
#include <dirent.h>//librería que se debe descargar de https://web.archive.org/web/20170428133315/http://www.softagalleria.net/dirent.php y seguir las instrucciones
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iomanip>

using namespace std;

// Espacio de nombres de OpenCV
using namespace cv;

//funciones
void listarArchivos();//consulta los archivos del directorio dir y crea un vetor con los nombres de esos archivos
int generarNumeros(int);//genera numeros aleatorios entre 0 y el número de archivos encontrados
std::vector<String> generarConjunto(int);//genera el conjunto de nombres de archivos para training y testing
void calcularMomentosTraining(std::vector<String>);//calcula y guarda los momentos de hu y el tipo de figura en el arreglo training
void calcularMomentosTest(std::vector<String>);//calcula y guarda los momentos de hu y el tipo de figura en el arreglo training
void listarMomentosTraining();//listar el vector training
void listarMomentosTest();//listar el vector test
int idFigura(String);//devuelve un número entero según la entrada de la forma 0=Circle, 2=Star, 3=Triangle, 4=Square...
string nombreFigura(int);//devuelve el nombre de la figura según el número ingresado de la forma similar a la función anterior

double calcularDistancia(Mat, Mat);
float compararImagenes();

void pruebaHu();
void analizarImagen();//muestra las distintas transformaciones de las imagenes
void calcularHistograma();
void calculoLBP();

//variables globales
const int ntraining=100;//cantidad de imágenes para el entrenamiento
const int ntest=75;//cantidad de imágenes para la validación
const int thresh = 1;//porcentaje para buscar bordes
double training[ntraining][8];//lista de hu e id de figuras de entrenamiento
double test[ntest][8];//lista de hu e id de figuras para testing
const std::string dir="C:/Users/EnrJ31022/Documents/IVC/Srwk_1/ShapeDataset/";//path en donde se encuentran los archivos
std::vector<String> lista, conjuntoEntrenamiento, conjuntoValidacion;//vectores que contendrán los nombres de archivos, entrenamiento, testing

int main(int argc, char** argv) {
    int opc=0;
    do{
        printf("\n%s \n", "PROGRAMA CLASIFICADOR");
        printf("%s \n", "1: Generar conjuntos de imagenes");
        printf("%s \n", "2: Calcular momentos de hu");
        printf("%s \n", "3: Ver momentos de entrenamiento");
        printf("%s \n", "4: Ver momentos de testing");
        printf("%s \n", "5: Ver analisis de imagenes");
        printf("%s \n", "6: Ver histograma de imagen");
        printf("%s \n", "7: Comparar imágenes");
        printf("%s \n", "8: Calculo LBP");
        printf("%s \n", "9: Salir");
        printf("%s \n", "Ingrese la opcion: ");
        scanf_s("%d", &opc);
        switch (opc){
        case 1:
            listarArchivos();
            conjuntoEntrenamiento=generarConjunto(ntraining);
            conjuntoValidacion=generarConjunto(ntest);
            break;
        case 2:
            calcularMomentosTraining(conjuntoEntrenamiento);
            calcularMomentosTest(conjuntoValidacion);
            printf("%s\n","CALCULADO CORRECTAMENTE");
            break;  
        case 3:
            listarMomentosTraining();
            break; 
        case 4:
            listarMomentosTest();
            break;
        case 5:
            analizarImagen();
            break;
        case 6:
            calcularHistograma();
            break;
        case 7:
            compararImagenes();
            break;
        case 8:
            calculoLBP();
            break;
        case 9:
            printf("%s \n", "Fin");
            break;     
        default:
            printf("%s \n","Valor desconocido");
            break;
        }
    }while(opc != 9);  

}

void listarArchivos(){
    int numero=0;
    std::string elem;
    DIR *direccion;
    struct dirent *elementos;
    if(lista.size() != 0)
        lista.clear();
    if(direccion=opendir(dir.c_str())){
        while(elementos=readdir(direccion)){
            lista.push_back(elementos->d_name);
            numero++;
        }
    }
    closedir(direccion);
    std::cout<<"Cantidad archivos encontrados: "<<numero<<"\n";

    std::cout<<"Elementos de la lista"<<"\n";
    std::cout <<lista.size() <<endl;
}

int generarNumeros(int ls){
    int a;
    a = rand() % (ls+1);
    return a;
}

std::vector<String> generarConjunto(int cantidad){
    std::vector<String> conjunto;
    for(int i=0;i<cantidad;i++){
        int posicion=generarNumeros((int)lista.size());
        conjunto.push_back(lista[posicion]);
    }
    return conjunto;
}

void calcularMomentosTraining(std::vector<String> conjunto){
        // Read image as grayscale image 
        for(int i=0;i<conjunto.size();i++){
            String filename = dir+conjunto[i];
            Mat image =imread(filename);
            Mat im;
            cvtColor(image,im,COLOR_BGR2GRAY);
    
            Mat canny_output;

            threshold(im,im,thresh,thresh*2,3);
            Canny( im, canny_output, thresh, thresh*2, 3 );
            Moments mu = moments(canny_output);

            double huMoments[7]; 
            HuMoments(mu, huMoments);          
            for(int j = 0; j < 7; j++)
                huMoments[j] = -1 * copysign(1.0, huMoments[j]) * log10(abs(huMoments[j])); //log transforma para tener valores no muy pequeños 
            for(int j=0;j<7;j++){
                training[i][j]=huMoments[j];
            }
            training[i][7]=idFigura(conjunto[i]);
        }       
}

void calcularMomentosTest(std::vector<String> conjunto){
        // Read image as grayscale image 
        for(int i=0;i<conjunto.size();i++){
            String filename = dir+conjunto[i];
            Mat image =imread(filename);
            Mat im;
            cvtColor(image,im,COLOR_BGR2GRAY);
    
            Mat canny_output;

            Canny( im, canny_output, thresh, thresh*2, 3 );
            Moments mu = moments(canny_output);

            double huMoments[7]; 
            HuMoments(mu, huMoments);          
            for(int j = 0; j < 7; j++)
                huMoments[j] = -1 * copysign(1.0, huMoments[j]) * log10(abs(huMoments[j])); 
            for(int j=0;j<7;j++){
                test[i][j]=huMoments[j];
            }
            test[i][7]=idFigura(conjunto[i]);
        }       
}

int idFigura(String fichero){
    int posi=(int)fichero.find("_");
    string nfigura=fichero.substr(0,posi);
    if(nfigura.compare("Circle")==0)
        return 0;
    else
        if (nfigura.compare("Heptagon")==0)
            return 7;
        else
            if (nfigura.compare("Hexagon")==0)
                return 6;
            else 
                if (nfigura.compare("Nonagon")==0)
                    return 9;
                else
                    if (nfigura.compare("Octagon")==0)
                        return 8;
                    else
                        if (nfigura.compare("Pentagon")==0)
                            return 5;
                        else
                            if (nfigura.compare("Square")==0)
                                return 4;
                            else
                                if (nfigura.compare("Star")==0)
                                    return 2;
                                else
                                    if (nfigura.compare("Triangle")==0)
                                        return 3;
    return 0;
}

void listarMomentosTraining(){
    printf("%s \n", "TRAINING: Momentos de hu e identificador");
    printf("%s      %s      %s      %s      %s      %s      %s      %s\n","h1","h2","h3","h4","h5","h6","h7", "id");
    for(int i=0;i<ntraining;i++){
        printf("%.2f      %.2f      %.2f      %.2f      %.2f      %.2f      %.2f      %s\n",training[i][0],training[i][1],training[i][2],training[i][3],training[i][4],training[i][5],training[i][6],nombreFigura((int)training[i][7]).c_str());
    }
}

void listarMomentosTest(){
    printf("%s \n", "TEST: Momentos de hu e identificador");
    printf("%s      %s      %s      %s      %s      %s      %s      %s\n","h1","h2","h3","h4","h5","h6","h7", "id");
    for(int i=0;i<ntest;i++){
        printf("%.2f      %.2f      %.2f      %.2f      %.2f      %.2f      %.2f      %s\n",test[i][0],test[i][1],test[i][2],test[i][3],test[i][4],test[i][5],test[i][6],nombreFigura((int)test[i][7]).c_str());
    }
}

string nombreFigura(int idFigura){
    switch (idFigura)
    {
    case 0:
        return "Circle";
        break;
    case 2:
        return "Star";
        break;
    case 3:
        return "Triangle";
        break;
    case 4:
        return "Square";
        break;
    case 5:
        return "Pentagon";
        break;
    case 6:
        return "Hexagon";
        break;
    case 7:
        return "Heptagon";
        break;
    case 8:
        return "Octagon";
        break;
    case 9:
        return "Nonagon";
        break;  
    default:
        return "Unknown";
        break;
    }
}

Mat src_gray;

void pruebaHu(){
    listarArchivos();
    for(int k=2;k<10;k++){
        std::cout<<dir+lista[k]+"\n";
        Mat src = imread(dir+lista[k]);
        cvtColor( src, src_gray, COLOR_BGR2GRAY );
        Mat canny_output;
        Canny( src_gray, canny_output, thresh, thresh*2, 3 );
        vector<vector<Point> > contours;
        findContours( canny_output, contours, RETR_TREE, CHAIN_APPROX_SIMPLE );
        vector<Point> contour;
        double area=0;
        for( size_t i = 0; i < contours.size(); i++ )
        {
            double auxarea = contourArea(contours[i]);
            if(auxarea > area) {
                area=auxarea;
                contour=contours[i];
            }
        }
        Moments m=moments(contour);
        double huMoments[7]; 
        HuMoments(m, huMoments);  
        std::cout<<"Hu moments: \n";
        for(int j = 0; j < 7; j++)
            huMoments[j] = -1 * copysign(1.0, huMoments[j]) * log10(abs(huMoments[j])); //log transforma para tener valores no muy pequeños 
        for(int i=0;i<7;i++){
            std::cout<<huMoments[i]<<"     ";
            }
        std::cout<<"\n";
    }   
}

//Mat src_gray;
//int thresh = 100;
RNG rng(12345);

void analizarImagen(){
        Mat src = imread(dir+lista[generarNumeros((int)lista.size())]);

        //mostrar imagen original        
        imshow( "Imagen", src );

        //transformar a escala de grises
        cvtColor( src, src_gray, COLOR_BGR2GRAY );
        blur( src_gray, src_gray, Size(3,3) );

        //mostrar imagen clon 
        imshow( "Clon", src_gray );

        //detección de bordes        
        Mat canny_output;
        Canny( src_gray, canny_output, thresh, thresh*2, 3 );
        vector<vector<Point> > contours;
        findContours( canny_output, contours, RETR_TREE, CHAIN_APPROX_SIMPLE );
        //graficar bordes
        Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
        Scalar color = Scalar( 255,255,255);
        drawContours( drawing, contours,1, color, 2 );
        //mostrar gráfica con bordes 
        imshow( "Shape", drawing );

        //calcular hu moments
        Moments m=moments(contours[0]);
        double huMoments[7]; 
        HuMoments(m, huMoments);  
        std::cout<<"Hu moments: \n";
        for(int j = 0; j < 7; j++)
            huMoments[j] = -1 * copysign(1.0, huMoments[j]) * log10(abs(huMoments[j])); //log transforma para tener valores no muy pequeños 
        for(int i=0;i<7;i++){
            std::cout<<huMoments[i]<<"     ";
            }
        std::cout<<"\n";
        waitKey();
}

void calcularHistograma(){
    Mat src = imread(dir+lista[generarNumeros((int)lista.size())]);
    vector<Mat> bgr_planes;
    split( src, bgr_planes );
    int histSize = 256;
    float range[] = { 0, 256 }; //the upper boundary is exclusive
    const float* histRange[] = { range };
    bool uniform = true, accumulate = false;
    Mat b_hist, g_hist, r_hist;
    calcHist( &bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, histRange, uniform, accumulate );
    calcHist( &bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, histRange, uniform, accumulate );
    calcHist( &bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, histRange, uniform, accumulate );
    int hist_w = 512, hist_h = 400;
    int bin_w = cvRound( (double) hist_w/histSize );
    Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );
    normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
    normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
    normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
    for( int i = 1; i < histSize; i++ )
    {
        line( histImage, Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ),
              Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
              Scalar( 255, 0, 0), 2, 8, 0  );
        line( histImage, Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ),
              Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
              Scalar( 0, 255, 0), 2, 8, 0  );
        line( histImage, Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ),
              Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
              Scalar( 0, 0, 255), 2, 8, 0  );
    }
    imshow("imagen de origen", src );
    imshow("Histograma", histImage );
    waitKey();

}
double calcularDistancia(Mat im1, Mat im2){
    return matchShapes(im1, im2, CONTOURS_MATCH_I2, 0); 
}
float compararImagenes(){
    String filename1 = dir+lista[generarNumeros((int)lista.size())];
    Mat im1 =imread(filename1);
    String filename2 = dir+lista[generarNumeros((int)lista.size())];
    Mat im2 =imread(filename2);
    

    cvtColor(im1,im1,COLOR_BGR2GRAY);   
    threshold(im1,im1,thresh,thresh*2,3);

    cvtColor(im2,im2,COLOR_BGR2GRAY);   
    threshold(im2,im2,thresh,thresh*2,3);

    //detección de bordes        
    Mat canny_output;
    Canny( im1, canny_output, thresh, thresh*2, 3 );
    vector<vector<Point> > contours;
    findContours( canny_output, contours, RETR_TREE, CHAIN_APPROX_SIMPLE );
    //graficar bordes
    Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
    Scalar color = Scalar( 255,255,255);
    drawContours( drawing, contours,1, color, 2 );

    //detección de bordes        
    Mat canny_output_;
    Canny( im2, canny_output_, thresh, thresh*2, 3 );
    vector<vector<Point> > contours_;
    findContours( canny_output_, contours_, RETR_TREE, CHAIN_APPROX_SIMPLE );
    //graficar bordes
    Mat drawing_ = Mat::zeros( canny_output_.size(), CV_8UC3 );
    
    drawContours( drawing_, contours_,1, color, 2 );

    imshow( "Im1", drawing );
    imshow( "Im2", drawing_);
    waitKey();

    double d = calcularDistancia(im1,im2);
    std::cout<<"Distancia : "<<d<<"\n";
    if(d>0.09){
        std::cout<<"Son figuras distintas"<<"\n";
        return true;
    }else{
        std::cout<<"Son figuras similares"<<"\n";
        return false;
    }
}

void calculoLBP(){
    String filename1 = dir+lista[generarNumeros((int)lista.size())];
    Mat im1 =imread(filename1);
    cvtColor(im1,im1,COLOR_BGR2GRAY);   
    threshold(im1,im1,thresh,thresh*2,3);
    Mat lbp=cv::Mat::zeros(im1.rows-2,im1.cols-2, CV_8UC1);
    for(int i=1;i<im1.rows-1;i++){
        for(int j=1;j<im1.cols-1;j++){
            unsigned char center = im1.at<unsigned char>(i,j);
            unsigned char code = 0;
            code |= (im1.at<unsigned char>(i-1,j-1) > center) << 7;
            code |= (im1.at<unsigned char>(i-1,j) > center) << 6;
            code |= (im1.at<unsigned char>(i-1,j+1) > center) << 5;
            code |= (im1.at<unsigned char>(i,j+1) > center) << 4;
            code |= (im1.at<unsigned char>(i+1,j+1) > center) << 3;
            code |= (im1.at<unsigned char>(i+1,j) > center) << 2;
            code |= (im1.at<unsigned char>(i+1,j-1) > center) << 1;
            code |= (im1.at<unsigned char>(i,j-1) > center) << 0;
            lbp.at<unsigned char>(i-1,j-1) = code;
        }
    }
    for(int i=0;i<lbp.cols;i++){
        for(int j=0;j<lbp.cols;j++){
            std::cout<<lbp.at<unsigned char>(i,j)<<" ";
        }
        // getchar();
        std::cout<<"\n";
    }
}