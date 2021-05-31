#ifndef WIDGET_H
#define WIDGET_H

#include <QWidget>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <QTimer>
using namespace cv;
using namespace std;
namespace Ui {
class Widget;
}

class Widget : public QWidget
{
    Q_OBJECT

public:
    explicit Widget(QWidget *parent = 0);
    ~Widget();
    void receive();
    void img_capture();
private slots:
    void on_Button1_clicked();

    void on_Button2_clicked();

    void on_Button3_clicked();

    void on_Button4_clicked();

    void on_Button5_clicked();

    void on_pushButton_clicked();

private:
    Ui::Widget *ui;
    Mat frame;
    VideoCapture vcapture;
    QTimer *timer;
    QTimer *timerc;
};

#endif // WIDGET_H
