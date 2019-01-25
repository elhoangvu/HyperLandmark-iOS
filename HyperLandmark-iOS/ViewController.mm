//
//  ViewController.m
//  HyperLandmark-iOS
//
//  Created by Le Hoang Vu on 1/25/19.
//  Copyright © 2019 Le Hoang Vu. All rights reserved.
//

#import "ViewController.h"

#import <opencv2/opencv.hpp>
#import <opencv2/videoio/cap_ios.h>

#include <iostream>
#include <vector>
#include <iostream>
#include <fstream>
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include "ldmarkmodel.h"

#include "opencv+parallel_for_.h"

using namespace std;
using namespace cv;

@interface ViewController () <CvVideoCameraDelegate> {
    ldmarkmodel* _modelt;
    std::vector<cv::Mat>* _currentShape;
}

@property (nonatomic, strong) CvVideoCamera* videoCamera;
@property (weak, nonatomic) IBOutlet UIView *cameraView;


@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    
    self.videoCamera = [[CvVideoCamera alloc] initWithParentView:self.cameraView];
    self.videoCamera.delegate = self;
    self.videoCamera.defaultAVCaptureDevicePosition = AVCaptureDevicePositionFront;
    self.videoCamera.defaultAVCaptureSessionPreset = AVCaptureSessionPreset640x480;
    self.videoCamera.defaultAVCaptureVideoOrientation = AVCaptureVideoOrientationPortrait;
    self.videoCamera.defaultFPS = 30;
    self.videoCamera.grayscaleMode = NO;
    
    NSString* haarPath = [NSBundle.mainBundle pathForResource:@"haar_facedetection" ofType:@"xml"];
    NSString* modelPath = [NSBundle.mainBundle pathForResource:@"landmark-model" ofType:@"bin"];
    _modelt = new ldmarkmodel([haarPath UTF8String]);
    
    if(!load_ldmarkmodel([modelPath UTF8String], *_modelt)) {
        std::cout << "Modle Opening Failed." << [modelPath UTF8String] << std::endl;
    }
    
    _currentShape = new std::vector<cv::Mat>(MAX_FACE_NUM);
}

- (IBAction)startCapture:(id)sender {
    [self.videoCamera start];
}

- (void)processImage:(cv::Mat &)image
{
    _modelt->track(image, *_currentShape);
    cv::Vec3d eav;
    _modelt->EstimateHeadPose((*_currentShape)[0], eav);
    _modelt->drawPose(image, (*_currentShape)[0], 50);
    parallel_for_(cv::Range(0, MAX_FACE_NUM), [&](const cv::Range& range){
        for (int i = range.start; i < range.end; i++){
            if (!(*_currentShape)[i].empty()){
                int numLandmarks = (*_currentShape)[i].cols / 2;
                for (int j = 0; j < numLandmarks; j++){
                    int x = (*_currentShape)[i].at<float>(j);
                    int y = (*_currentShape)[i].at<float>(j + numLandmarks);
                    cv::circle(image, cv::Point(x, y), 2, cv::Scalar(0, 0, 255), -1);
                }
            }
        }
    });
}


@end
