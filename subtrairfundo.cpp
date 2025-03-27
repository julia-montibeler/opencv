#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace std;

using namespace cv;

int main() {
    // Open the default camera (0)
    cv::VideoCapture cap(0);

    // Check if the camera is opened successfully
    if (!cap.isOpened()) {
        std::cout << "Error: Could not open video capture." << std::endl;
        return -1;
    }

    // Declare a frame variable to store the captured frame
    cv::Mat frame, fundo, img;
    cap >> fundo;

	img = fundo.clone();


    // Capture and display video until 'Esc' key is pressed
    while (true) {
        // Capture a new frame from the camera
        cap >> frame;

	    for (int i = 0; i < frame.rows; i++)
		    for (int j = 0; j < frame.cols; j++)
            {
                cv::Vec3b framergb = frame.at<cv::Vec3b>(i, j);
                cv::Vec3b fundorgb = fundo.at<cv::Vec3b>(i, j);

                if  ((abs((int)framergb[0] - (int)fundorgb[0] ) > 20) || 
                    (abs((int)framergb[1] - (int)fundorgb[1]) > 20) ||
                    (abs((int)framergb[2] - (int)fundorgb[2]) > 20))
                        img.at<Vec3b>(i, j) = frame.at<Vec3b>(i, j);
                else 
                    img.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
                    
            }
			

        // Check if the frame is empty (in case of an error)
        if (frame.empty()) {
            std::cout << "Error: Failed to capture frame." << std::endl;
            break;
        }

        // Show the captured frame in a window
        cv::imshow("Video Capture", img);

        //Wait for 1 ms and check if the 'Esc' key (ASCII 27) is pressed
        if (cv::waitKey(1) == 27) {
            break;  // Exit the loop if 'Esc' is pressed
        }

        if (cv::waitKey(1) == 'a') {
            std::cout << "capturando" << std::endl;
            cap >> fundo;
        }

        for (int i = 0; i < frame.rows; i++)
		    for (int j = 0; j < frame.cols; j++)
            {
                float a = 0.1;
                cv::Vec3b rgbframe = frame.at<cv::Vec3b>(i, j);
                cv::Vec3b rgbfundo = fundo.at<cv::Vec3b>(i, j);
                float r = (1-a)*rgbfundo[0] + a*rgbframe[0];
                float g = (1-a)*rgbfundo[1] + a*rgbframe[1];
                float b = (1-a)*rgbfundo[2] + a*rgbframe[2];
                fundo.at<cv::Vec3b>(i, j) = cv::Vec3b((char)r, (char)g, (char)b);
                    
            }
    }

    // Release the camera and close all OpenCV windows
    cap.release();
    cv::destroyAllWindows();

    return 0;
}
