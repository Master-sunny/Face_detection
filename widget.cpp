#include "widget.h"
#include "ui_widget.h"
#include <iostream>
#include "string.h"
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <QFileDialog>
#include <fstream>  //使用ifstream 的头文件
#include <QImage>
#include <QDebug>
#include <QTimer>
using namespace std;
using namespace cv::face;
using namespace cv;

string haar_face_datapath = "D:/OpencvSouce/opencv32/build/etc/haarcascades/haarcascade_frontalface_alt_tree.xml";
string valueline, path, classlabel;
string cap_img_path="D:/Myfaces/face2_%d.jpg";
QString num_str;
vector<int> labels;
vector<Mat> images;
vector<int> test_labels;
vector<Mat> test_images;
char separator = ';';
vector<Rect> faces;
vector<Rect> capture_faces;
int testlabel;
Ptr<BasicFaceRecognizer> model = createEigenFaceRecognizer();
Mat testSample;
Mat testSample_cptest;
int testLabel_cptest;
int testLabel;
Mat temp;
int img_count;
CascadeClassifier faceDetector;
Widget::Widget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::Widget)
{
    ui->setupUi(this);
    timer=new QTimer(this);
    timerc=new QTimer(this);
    connect(timer,&QTimer::timeout,this,&Widget::receive);
    connect(timerc,&QTimer::timeout,this,&Widget::img_capture);
    string filename = string("D:/Myfaces/images1.csv");
        ifstream file(filename.c_str(), ifstream::in);

        if (!file) {
            ui->line2->setText("invalid load");
        }
        else{
            //遍历CSV文件里的路径，用来给训练器提供路径
            //读取每一行
            while (getline(file, valueline)) {
                stringstream liness(valueline);
                getline(liness, path, separator);
                getline(liness, classlabel);
                if (!path.empty() && !classlabel.empty()) {
                    //printf("path : %s\n", path.c_str());
                    images.push_back(imread(path, 0));
                    labels.push_back(atoi(classlabel.c_str()));
                }
            }
            //没有图片则停止
            if (images.size() < 1 || labels.size() < 1) {
                ui->line2->setText("no picture");
            }
            else{
                //获取图片的大小
                int height = images[0].rows;
                int width = images[0].cols;
                //printf("height : %d, width : %d\n", height, width);
                //测试图片，用最后一张用来检测是否正确
                testSample = images[images.size()-1];
                testLabel = labels[images.size()-1];
                //imshow("test", testSample);
                //printf("size=%d\n", images.size());
                //printf("height : %d, width : %d\n", testSample.cols, testSample.rows);
                //把图片拿出去
                images.pop_back();
                labels.pop_back();
                // Eigen训练图片
                model->train(images, labels);

                // 识别
                int predictedLabel = model->predict(testSample);
                if(testLabel==predictedLabel){
                    ui->line2->setText("pretest sucess");
                }
                else{
                    ui->line2->setText("pretest wrong");
                }
            }
        }
}

Widget::~Widget()
{
    delete ui;
}

void Widget::receive()
{
    //人脸识别的分类器，这里使用haar级联分类器找到人脸位置
    faceDetector.load(haar_face_datapath);
    QString s ;
    vcapture.read(frame);
    //printf("h=%d,w=%d\n",frame.rows,frame.cols);
    flip(frame, frame, 1);
    Mat dst;
    faceDetector.detectMultiScale(frame, faces, 1.1, 1, 0, Size(80, 100), Size(380, 400));
    if(faces.size()==0)
    {
        s="xxxxxxxx";
        ui->line1->setText(s);
    }
    else{
        for (int i = 0; i < faces.size(); i++) {
            Mat roi = frame(faces[i]);
            cvtColor(roi, dst, COLOR_BGR2GRAY);
            cv::resize(dst, testSample, Size(100, 100));
            //面部识别获得标签
            testlabel= model->predict(testSample);
            rectangle(frame, faces[i], Scalar(255, 0, 0), 2, 8, 0);
            if(testlabel>=15083201 && testlabel<=15083236){
                s = QString::number(testlabel,10);
                putText(frame,  "class:150832", faces[i].tl(), FONT_HERSHEY_PLAIN, 1.0, Scalar(0, 0, 255), 2, 8);
            }
            else{
                s="xxxxxxxx";
                putText(frame,  "unknown", faces[i].tl(), FONT_HERSHEY_PLAIN, 1.0, Scalar(0, 0, 255), 2, 8);
            }
             ui->line1->setText(s);
        }
    }
    cvtColor(frame,temp,CV_BGR2RGB);
    QImage img = QImage( (const unsigned char*)(temp.data), temp.cols, temp.rows, QImage::Format_RGB888 );
    ui->label1->setPixmap(QPixmap::fromImage(img));
}
void Widget::on_Button1_clicked()//打开摄像头
{
    vcapture.open(0);
    if(!vcapture.isOpened()){
        ui->line2->setText("Camera cannot open");
    }
    timer->start(40);
}

void Widget::on_Button2_clicked()//close
{
    this->close();
}

void Widget::on_Button3_clicked()//采集人脸
{
    vcapture.open(0);
    if(!vcapture.isOpened()){
        ui->line2->setText("Camera cannot open");
    }
    else{
        timerc->start(40);
    }
}
void Widget::img_capture()
{
    vcapture.read(frame);
    flip(frame,frame,1);
    faceDetector.load(haar_face_datapath);
    faceDetector.detectMultiScale(frame,capture_faces,1.1,1,0,Size(80, 100), Size(380, 400));
    for(int i=0;i<capture_faces.size();i++){
        if (img_count % 2 == 0) {
            Mat dst;
            cv::resize(frame(capture_faces[i]), dst, Size(100, 100));
            imwrite(format(cap_img_path.c_str(), img_count), dst);
            cvtColor(dst,dst,CV_BGR2RGB);
            cv::resize(dst, dst, Size(200, 200));
            QImage img1 = QImage( (const unsigned char*)(dst.data), dst.cols, dst.rows, QImage::Format_RGB888 );
            ui->label2->setPixmap(QPixmap::fromImage(img1));
        }
        rectangle(frame,capture_faces[i],Scalar(0,0,255),2,8,0);
    }
    img_count++;
    cvtColor(frame,temp,CV_BGR2RGB);
    QImage img = QImage( (const unsigned char*)(temp.data), temp.cols, temp.rows, QImage::Format_RGB888 );
    ui->label1->setPixmap(QPixmap::fromImage(img));

}

void Widget::on_Button4_clicked()
{
    img_count=0;
    ui->label2->setText("        finish");
    timerc->stop();
}

void Widget::on_Button5_clicked()//检测采集结果
{
    Mat dst;
    QImage img1;
    string s =num_str.toStdString();
    string filename = string("D:/Myfaces/images.csv");
        ifstream file(filename.c_str(), ifstream::in);

        if (!file) {
            ui->line2->setText("invalid load");
        }
        else{
            //遍历CSV文件里的路径，用来给训练器提供路径
            //读取每一行
            while (getline(file, valueline)) {
                stringstream liness(valueline);
                getline(liness, path, separator);
                getline(liness, classlabel);
                if (!path.empty() && !classlabel.empty()) {
                    if(strstr(path.c_str(),s.c_str()))
                    {
                        test_images.push_back(imread(path,0));
                        test_labels.push_back(atoi(classlabel.c_str()));
                    }
                }
                else{
                    ui->line2->setText("csvfile is empty");
                }
            }
            //没有图片则停止
            if (test_images.size() < 1 || test_labels.size() < 1) {
                ui->line2->setText("no picture");
            }
            else{
                imshow("test", test_images[0]);
                // Eigen训练图片
                model->train(test_images, test_labels);
                // 识别
                testLabel=test_labels[0];
                int predictedLabel = model->predict(test_images[0]);
                if(testLabel==predictedLabel){
                    Mat mean = model->getMean();
                    Mat meanFace = mean.reshape(1, test_images[0].rows);
                    if (meanFace.channels() == 1) {
                         //0到255均衡化
                        normalize(meanFace, dst, 0, 255, NORM_MINMAX, CV_8UC1);
                        cv::resize(dst,dst,Size(200,200));
                        img1 = QImage( (const unsigned char*)(dst.data), dst.cols, dst.rows, QImage::Format_Indexed8 );
                    }
                     else if (meanFace.channels() == 3) {
                            //0到255均衡化
                        normalize(meanFace, dst, 0, 255, NORM_MINMAX, CV_8UC3);
                        cvtColor(dst,dst,CV_BGR2RGB);
                        cv::resize(dst,dst,Size(200,200));
                        img1 = QImage( (const unsigned char*)(dst.data), dst.cols, dst.rows, QImage::Format_RGB888 );
                    }

                    ui->label2->setPixmap(QPixmap::fromImage(img1));
                    ui->line2->setText("test sucess");
                }
                else{
                    ui->line2->setText("test wrong");
                }
            }
        }
}

void Widget::on_pushButton_clicked()//拼接路径
{
   num_str = ui->textEdit->toPlainText();
   QString temppath="D:/Myfaces/face"+num_str+"_%d.jpg";
   cap_img_path=temppath.toStdString();
}
