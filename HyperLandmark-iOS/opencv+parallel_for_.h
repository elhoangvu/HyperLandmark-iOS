//
//  opencv+parallel_for_.h
//  HyperLandmark-iOS
//
//  Created by Le Hoang Vu on 1/25/19.
//  Copyright © 2019 Le Hoang Vu. All rights reserved.
//

#ifndef opencv_parallel_for__h
#define opencv_parallel_for__h

#import <opencv2/opencv.hpp>

class ParallelLoopBodyLambdaWrapper : public cv::ParallelLoopBody
{
private:
    std::function<void(const cv::Range&)> m_functor;
public:
    ParallelLoopBodyLambdaWrapper(std::function<void(const cv::Range&)> functor) :
    m_functor(functor)
    { }
    
    virtual void operator() (const cv::Range& range) const
    {
        m_functor(range);
    }
};

inline void parallel_for_(const cv::Range& range, std::function<void(const cv::Range&)> functor, double nstripes=-1.)
{
    parallel_for_(range, ParallelLoopBodyLambdaWrapper(functor), nstripes);
}

#endif /* opencv_parallel_for__h */
