/*****************************************************************************
*   Changes by Motiejus Ëringis
*   Original header below.
******************************************************************************
*   Number Plate Recognition using SVM and Neural Networks
******************************************************************************
*   by David Millán Escrivá, 5th Dec 2012
*   http://blog.damiles.com
******************************************************************************
*   Ch5 of the book "Mastering OpenCV with Practical Computer Vision Projects"
*   Copyright Packt Publishing 2012.
*   http://www.packtpub.com/cool-projects-with-opencv/book
*****************************************************************************/

// Main entry code OpenCV

#define SPEED_UPDATE_INTERVAL 60 // Frames
#define PLATE_TIMEOUT 250 // Frames
#define FOCAL_LENGTH 1

#include <cv.h>
#include <highgui.h>
#include <cvaux.h>
#include <ml.h>

#include <iostream>
#include <vector>

#include "DetectRegions.h"
#include "OCR.h" /* TODO: make OCR work with LT plates or remove */

using namespace std;
using namespace cv;

/* TODO: remove getFilename? */
string getFilename(string s) {

    char sep = '/';
    char sepExt='.';

    #ifdef _WIN32
        sep = '\\';
    #endif

    size_t i = s.rfind(sep, s.length( ));
    if (i != string::npos) {
        string fn= (s.substr(i+1, s.length( ) - i));
        size_t j = fn.rfind(sepExt, fn.length( ));
        if (i != string::npos) {
            return fn.substr(0,j);
        }else{
            return fn;
        }
    }else{
        return "";
    }
}

struct TrackedObj {
    Rect lastPos;
    Rect currPos;
    int    lastUpd;
    double speed;
};

int main(int argc, char* argv[])
{
    // Check the number of parameters
    if (argc < 3) {
        std::cerr << "Naudojimas: [pradinis video] [isvedamas video]" << std::endl;
        return 1;
    }

    std::cout.precision(2);
    VideoCapture cap(argv[1]); // open the video file for reading

    if(!cap.isOpened())  // if not success, exit program
    {
        cerr << "Nepavyko atidaryti pradinio video failo \"" << argv[1] << "\"." << endl;
        return -1;
    }

    double fps = cap.get(CV_CAP_PROP_FPS); //get the frames per seconds of the video

    VideoWriter output_cap(argv[2], 
               //cap.get(CV_CAP_PROP_FOURCC),
               -1,
               cap.get(CV_CAP_PROP_FPS),
               cv::Size(cap.get(CV_CAP_PROP_FRAME_WIDTH),
               cap.get(CV_CAP_PROP_FRAME_HEIGHT)));

    if (!output_cap.isOpened())
    {
        cerr << "Nepavyko atidaryti isvedimo video failo \"" << argv[2] << "\"." << endl;
        return -1;
    }

    cout << "Kadrai per sekunde : " << fps << endl;
    cout << "Iseiti is programos: [ESC]" << endl;

    namedWindow("MyVideo",CV_WINDOW_AUTOSIZE); //create a window called "MyVideo"
    std::string kadruSkaicius = std::to_string(int(cap.get(CV_CAP_PROP_FRAME_COUNT)));

    //SVM for each plate region to get valid car plates
    //Read file storage.
    FileStorage fs;
    fs.open("SVM.xml", FileStorage::READ); /* TODO: SVM needs retraining */
    Mat SVM_TrainingData;
    Mat SVM_Classes;
    fs["TrainingData"] >> SVM_TrainingData;
    fs["classes"] >> SVM_Classes;
    //Set SVM params
    CvSVMParams SVM_params;
    SVM_params.svm_type = CvSVM::C_SVC;
    SVM_params.kernel_type = CvSVM::LINEAR; //CvSVM::LINEAR;
    SVM_params.degree = 0;
    SVM_params.gamma = 1;
    SVM_params.coef0 = 0;
    SVM_params.C = 1;
    SVM_params.nu = 0;
    SVM_params.p = 0;
    SVM_params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 1000, 0.01);
    //Train SVM
    CvSVM svmClassifier(SVM_TrainingData, SVM_Classes, Mat(), Mat(), SVM_params);

    vector<TrackedObj> trackedObjects;
    while(1)
    {
        Mat input_image;
        Mat frame;
        std::string einamasisKadras = std::to_string(int(cap.get(CV_CAP_PROP_POS_FRAMES)));

        bool bSuccess = cap.read(input_image); // read a new frame from video
        
        if (!bSuccess) //if not success, break loop
        {
            cout << "Nepavyko atidaryti failo" << endl;
            break;
        }

        DetectRegions detectRegions;
        vector<Plate> posible_regions= detectRegions.run( input_image );

        //For each possible plate, classify with svm if it's a plate or no
        vector<Plate> plates;
        for(size_t i = 0; i < posible_regions.size(); i++)
        {
            Mat img=posible_regions[i].plateImg;
            Mat p= img.reshape(1, 1);
            p.convertTo(p, CV_32FC1);

            int response = (int)svmClassifier.predict( p );
            if(response==1)
                plates.push_back(posible_regions[i]);
        }

        for(size_t  i = 0; i < plates.size(); i++){
            bool foundMatching = false;
            Plate plate=plates[i];
            size_t j = 0;
            TrackedObj oldTrackedObj;

            for(; j < trackedObjects.size() && !foundMatching; j++) {
                Rect blueRect;
                blueRect = plate.position & trackedObjects[j].currPos;  // grazina staciakampiu sankirta
                if (blueRect.width != 0 || plate.position == trackedObjects[j].currPos) {
                    foundMatching = true;
                    oldTrackedObj = trackedObjects[j];
                    if (trackedObjects[j].lastUpd > SPEED_UPDATE_INTERVAL) {
						/* TODO: fix speed calculation */
                        trackedObjects[j].speed = double((plate.position.width - trackedObjects[j].lastPos.width)) / double(trackedObjects[j].lastUpd);
                        trackedObjects[j].lastUpd = 0;
                        trackedObjects[j].lastPos = plate.position;
                    }
                    trackedObjects[j].currPos = plate.position;
                }
            }
            if (!foundMatching) {
                TrackedObj newTrackedObj;
                newTrackedObj.lastUpd = 0;
                newTrackedObj.speed   = 0;
                newTrackedObj.lastPos = plate.position;
                newTrackedObj.currPos = plate.position;
                trackedObjects.push_back(newTrackedObj);
                j++;
                oldTrackedObj = newTrackedObj;
            }

            rectangle(input_image, plate.position, Scalar(0,0,200), 2);
            
            // putText(input_image, licensePlate, Point(plate.position.x, plate.position.y), CV_FONT_HERSHEY_SIMPLEX, 1, Scalar(0,0,200),2);
        }

        for(size_t  j = 0; j < trackedObjects.size(); j++) {
            trackedObjects[j].lastUpd++;
            rectangle(input_image, trackedObjects[j].currPos, Scalar(200,0,0));
            putText(input_image, "ID: " + std::to_string(j) + " V: " + std::to_string(trackedObjects[j].speed),
                Point(trackedObjects[j].currPos.x, trackedObjects[j].currPos.y), CV_FONT_HERSHEY_SIMPLEX, 1, Scalar(200, 0, 0), 2);
			if (trackedObjects[j].lastUpd > PLATE_TIMEOUT)
				trackedObjects.erase(trackedObjects.begin() + j);
        }

        cout << "Kadras " + einamasisKadras + "/" + kadruSkaicius << endl;
        putText(input_image, "Kadras " + einamasisKadras + "/" + kadruSkaicius, Point(25, 30), CV_FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 200), 2);
        imshow("MyVideo", input_image); //show the frame in "MyVideo" window

        output_cap.write(input_image);

        if(waitKey(30) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
        {
            cout << "[ESC] nuspaustas." << endl;
            break; 
        }
    }

    output_cap.release();
    cap.release();

    return 0;
}
