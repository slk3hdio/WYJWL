#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class BasketballShotAnalyzer {
private:
    int shotsDetected;
    int shotsMade;
    bool processingShot;
    cv::Point hoopCenter;
    int hoopRadius;
    std::vector<cv::Point> ballTrajectory;
    std::vector<cv::Point> previousBallPositions;
    std::vector<int> shotResults;

    // 改进的参数
    int minBallRadius;
    int maxBallRadius;
    double ballDetectionConfidenceThreshold;
    int trajectoryBufferSize;
    int minTrajectoryPointsForShot;
    double minMovementThreshold;
    int maxMissingFrames;
    int missingFrameCount;
    cv::Point lastValidBallPos;

    // 背景减除器
    cv::Ptr<cv::BackgroundSubtractor> backgroundSubtractor;
    bool useBackgroundSubtraction;

public:
    BasketballShotAnalyzer() :
        shotsDetected(0),
        shotsMade(0),
        processingShot(false),
        minBallRadius(8),
        maxBallRadius(40),
        ballDetectionConfidenceThreshold(0.5),
        trajectoryBufferSize(20),
        minTrajectoryPointsForShot(8),
        minMovementThreshold(5.0),
        maxMissingFrames(5),
        missingFrameCount(0),
        lastValidBallPos(-1, -1),
        useBackgroundSubtraction(true)
    {
        // 创建背景减除器
        backgroundSubtractor = cv::createBackgroundSubtractorMOG2(500, 16, true);
    }

    void setupHoop(const cv::Point& center, int radius) {
        hoopCenter = center;
        hoopRadius = radius;
        std::cout << "篮筐设置完成 - 中心: (" << center.x << ", " << center.y
            << "), 半径: " << radius << std::endl;
    }

    // 改进的球检测算法
    cv::Point detectBall(const cv::Mat& frame) {
        cv::Mat processFrame = frame.clone();

        // 方法1：基于颜色的检测（橙色球）
        cv::Point colorBasedBall = detectBallByColor(processFrame);

        // 方法2：基于背景减除的检测
        cv::Point motionBasedBall = detectBallByMotion(processFrame);

        // 方法3：基于圆形检测
        cv::Point shapeBasedBall = detectBallByShape(processFrame);

        // 融合多种检测结果
        cv::Point finalBall = fuseBallDetections(colorBasedBall, motionBasedBall, shapeBasedBall);

        // 使用简单的位置验证和平滑过滤
        return smoothBallPosition(finalBall);
    }

    // 更新过滤并检测投篮
    void updateTrajectory(const cv::Point& ballPos, cv::Mat& displayFrame) {
        // 如果找到了球
        if (ballPos.x >= 0 && ballPos.y >= 0) {
            missingFrameCount = 0;

            // 检测移动是否足够大（避免噪声）
            if (lastValidBallPos.x >= 0) {
                double movement = cv::norm(ballPos - lastValidBallPos);
                if (movement < minMovementThreshold && !processingShot) {
                    return; // 移动太小，可能是噪声
                }
            }

            // 更新位置历史
            previousBallPositions.push_back(ballPos);
            if (static_cast<int>(previousBallPositions.size()) > trajectoryBufferSize) {
                previousBallPositions.erase(previousBallPositions.begin());
            }

            lastValidBallPos = ballPos;

            // 显示球的位置
            cv::circle(displayFrame, ballPos, 8, cv::Scalar(0, 255, 255), -1);

            // 检测新的投篮开始
            if (!processingShot && previousBallPositions.size() >= 5) {
                if (detectShotStart()) {
                    processingShot = true;
                    ballTrajectory.clear();
                    std::cout << "检测到新的投篮！" << std::endl;
                }
            }

            // 如果正在跟踪投篮，记录过滤
            if (processingShot) {
                ballTrajectory.push_back(ballPos);
                checkShotCompletion(displayFrame);
            }
        }
        else {
            missingFrameCount++;

            // 如果正在处理投篮且丢失帧数不太多，继续处理
            if (processingShot && missingFrameCount <= maxMissingFrames) {
                if (!ballTrajectory.empty()) {
                    // 使用最后一个已知位置
                    ballTrajectory.push_back(ballTrajectory.back());
                }
                checkShotCompletion(displayFrame);
            }
            else if (missingFrameCount > maxMissingFrames && processingShot) {
                // 丢失太多帧，结束当前投篮跟踪
                processingShot = false;
                ballTrajectory.clear();
                std::cout << "跟踪丢失，重置投篮检测" << std::endl;
            }
        }
    }

private:
    // 基于颜色检测的球
    cv::Point detectBallByColor(const cv::Mat& frame) {
        cv::Mat hsvFrame;
        cv::cvtColor(frame, hsvFrame, cv::COLOR_BGR2HSV);

        // 两个橙色范围，适用于不同光照条件
        cv::Scalar lowerOrange1(0, 120, 70);
        cv::Scalar upperOrange1(10, 255, 255);
        cv::Scalar lowerOrange2(170, 120, 70);
        cv::Scalar upperOrange2(180, 255, 255);

        cv::Mat mask1, mask2, mask;
        cv::inRange(hsvFrame, lowerOrange1, upperOrange1, mask1);
        cv::inRange(hsvFrame, lowerOrange2, upperOrange2, mask2);
        cv::bitwise_or(mask1, mask2, mask);

        // 形态学操作
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);

        return findBestCircularContour(mask);
    }

    // 基于运动检测的球
    cv::Point detectBallByMotion(const cv::Mat& frame) {
        if (!useBackgroundSubtraction) return cv::Point(-1, -1);

        cv::Mat fgMask;
        backgroundSubtractor->apply(frame, fgMask);

        // 去除噪声
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::morphologyEx(fgMask, fgMask, cv::MORPH_OPEN, kernel);

        return findBestCircularContour(fgMask);
    }

    // 基于形状检测的球
    cv::Point detectBallByShape(const cv::Mat& frame) {
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // 高斯模糊
        cv::GaussianBlur(gray, gray, cv::Size(9, 9), 2, 2);

        // 霍夫圆检测
        std::vector<cv::Vec3f> circles;
        cv::HoughCircles(gray, circles, cv::HOUGH_GRADIENT, 1,
            static_cast<double>(gray.rows) / 8, 100, 30, minBallRadius, maxBallRadius);

        if (!circles.empty()) {
            // 选择最可能的圆
            cv::Vec3f bestCircle = circles[0];
            for (const auto& circle : circles) {
                if (lastValidBallPos.x >= 0) {
                    cv::Point currentPos(static_cast<int>(circle[0]), static_cast<int>(circle[1]));
                    cv::Point bestPos(static_cast<int>(bestCircle[0]), static_cast<int>(bestCircle[1]));

                    double currentDist = cv::norm(currentPos - lastValidBallPos);
                    double bestDist = cv::norm(bestPos - lastValidBallPos);

                    if (currentDist < bestDist) {
                        bestCircle = circle;
                    }
                }
            }
            return cv::Point(static_cast<int>(bestCircle[0]), static_cast<int>(bestCircle[1]));
        }

        return cv::Point(-1, -1);
    }

    // 在掩码中找到最好的圆形轮廓
    cv::Point findBestCircularContour(const cv::Mat& mask) {
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        cv::Point ballCenter(-1, -1);
        double bestScore = 0.0;

        for (const auto& contour : contours) {
            double area = cv::contourArea(contour);
            if (area < 50 || area > 5000) continue;

            double perimeter = cv::arcLength(contour, true);
            double circularity = 4 * M_PI * area / (perimeter * perimeter);

            cv::Point2f center;
            float radius;
            cv::minEnclosingCircle(contour, center, radius);

            // 综合评分：圆形度 + 大小适中 + 位置连续性
            double sizeScore = 1.0;
            if (radius < minBallRadius || radius > maxBallRadius) {
                sizeScore = 0.5;
            }

            double continuityScore = 1.0;
            if (lastValidBallPos.x >= 0) {
                double distance = cv::norm(cv::Point(static_cast<int>(center.x), static_cast<int>(center.y)) - lastValidBallPos);
                if (distance > 100) continuityScore = 0.3;
                else if (distance > 50) continuityScore = 0.7;
            }

            double totalScore = circularity * sizeScore * continuityScore;

            if (totalScore > bestScore && circularity > 0.3) {
                bestScore = totalScore;
                ballCenter = cv::Point(static_cast<int>(center.x), static_cast<int>(center.y));
            }
        }

        return ballCenter;
    }

    // 融合多种检测结果
    cv::Point fuseBallDetections(const cv::Point& colorBall,
        const cv::Point& motionBall,
        const cv::Point& shapeBall) {
        std::vector<cv::Point> validDetections;

        if (colorBall.x >= 0) validDetections.push_back(colorBall);
        if (motionBall.x >= 0) validDetections.push_back(motionBall);
        if (shapeBall.x >= 0) validDetections.push_back(shapeBall);

        if (validDetections.empty()) {
            return cv::Point(-1, -1);
        }

        // 如果只有一个检测结果
        if (validDetections.size() == 1) {
            return validDetections[0];
        }

        // 计算平均位置或选择最接近上一个位置的
        if (lastValidBallPos.x >= 0) {
            cv::Point bestDetection = validDetections[0];
            double minDistance = cv::norm(bestDetection - lastValidBallPos);

            for (const auto& detection : validDetections) {
                double distance = cv::norm(detection - lastValidBallPos);
                if (distance < minDistance) {
                    minDistance = distance;
                    bestDetection = detection;
                }
            }
            return bestDetection;
        }

        // 计算平均位置
        int avgX = 0, avgY = 0;
        for (const auto& detection : validDetections) {
            avgX += detection.x;
            avgY += detection.y;
        }
        return cv::Point(avgX / static_cast<int>(validDetections.size()),
            avgY / static_cast<int>(validDetections.size()));
    }

    // 平滑球的位置
    cv::Point smoothBallPosition(const cv::Point& detectedBall) {
        if (detectedBall.x < 0) return detectedBall;

        // 简单的平滑：如果与上一个位置距离太大，进行插值
        if (lastValidBallPos.x >= 0) {
            double distance = cv::norm(detectedBall - lastValidBallPos);
            if (distance > 80 && !processingShot) {
                // 距离太大，可能是误检，返回无效位置
                return cv::Point(-1, -1);
            }
        }

        return detectedBall;
    }

    // 检测投篮开始
    bool detectShotStart() {
        if (previousBallPositions.size() < 5) return false;

        // 检测球是否在向上运动
        int recentFrames = 5;
        int upwardMovement = 0;

        for (int i = previousBallPositions.size() - recentFrames;
            i < static_cast<int>(previousBallPositions.size()) - 1; i++) {
            if (previousBallPositions[i].y > previousBallPositions[i + 1].y) {
                upwardMovement++;
            }
        }

        // 如果大部分帧都显示向上运动，认为是投篮开始
        return upwardMovement >= 3;
    }

    // 检测投篮完成
    void checkShotCompletion(cv::Mat& displayFrame) {
        if (static_cast<int>(ballTrajectory.size()) < minTrajectoryPointsForShot) return;

        // 检测过滤是否通过篮筐
        bool passedThroughHoop = false;
        bool isDescending = false;
        int pointsNearHoop = 0;

        // 检测最近几个点的情况
        int recentPoints = std::min(8, static_cast<int>(ballTrajectory.size()));
        if (recentPoints >= 3) {
            int startIdx = ballTrajectory.size() - recentPoints;
            isDescending = ballTrajectory[startIdx].y < ballTrajectory[ballTrajectory.size() - 1].y;
        }

        // 计算有多少过滤点接近篮筐
        for (const auto& point : ballTrajectory) {
            double distance = cv::norm(point - hoopCenter);
            if (distance < hoopRadius * 1.5) {
                pointsNearHoop++;
            }
        }

        // 判断是否通过篮筐
        if (pointsNearHoop > 2) {
            passedThroughHoop = true;
        }

        // 如果球已经通过篮筐且开始下降，完成投篮
        if (passedThroughHoop && isDescending) {
            shotsDetected++;

            // 判断是否投进
            bool isMade = determineIfShotMade();
            if (isMade) {
                shotsMade++;
                shotResults.push_back(1);
                std::cout << "投篮 #" << shotsDetected << ": 投进了！" << std::endl;
            }
            else {
                shotResults.push_back(0);
                std::cout << "投篮 #" << shotsDetected << ": 未投进！" << std::endl;
            }

            // 重置状态，准备下一次投篮
            processingShot = false;
            ballTrajectory.clear();

            // 在画面上显示投篮结果
            std::string resultText = "投篮 #" + std::to_string(shotsDetected) +
                ": " + (isMade ? "进球" : "未进");
            cv::putText(displayFrame, resultText, cv::Point(50, 50 + shotsDetected * 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.7,
                isMade ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), 2);
        }
    }

    // 通过分析过滤判断球是否进入篮筐
    bool determineIfShotMade() {
        // 计算有多少点落在篮筐内
        int pointsInHoop = 0;
        for (const auto& point : ballTrajectory) {
            double distance = cv::norm(point - hoopCenter);
            if (distance < hoopRadius * 0.8) {
                pointsInHoop++;
            }
        }

        // 检测过滤末端是否在篮筐下方
        bool ballPassedThroughHoop = false;
        if (!ballTrajectory.empty()) {
            int endIdx = ballTrajectory.size() - 1;
            int startIdx = std::max(0, endIdx - 5);

            for (int i = startIdx; i <= endIdx; i++) {
                const cv::Point& point = ballTrajectory[i];
                if (std::abs(point.x - hoopCenter.x) < hoopRadius &&
                    point.y > hoopCenter.y &&
                    point.y < hoopCenter.y + hoopRadius * 2) {
                    ballPassedThroughHoop = true;
                    break;
                }
            }
        }

        return pointsInHoop >= 2 && ballPassedThroughHoop;
    }

public:
    // 统计信息显示方法
    void drawStats(cv::Mat& frame) {
        // 计算命中率
        double shootingPercentage = shotsDetected > 0 ?
            static_cast<double>(shotsMade) / shotsDetected * 100.0 : 0.0;

        // 显示篮筐
        cv::circle(frame, hoopCenter, hoopRadius, cv::Scalar(0, 165, 255), 2);

        // 显示统计信息
        std::string statsText = "检测到的投篮: " + std::to_string(shotsDetected) +
            ", 投进: " + std::to_string(shotsMade) +
            " (" + std::to_string(static_cast<int>(shootingPercentage)) + "%)";

        cv::putText(frame, statsText, cv::Point(10, 30),
            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);

        // 显示每次投篮的结果
        for (size_t i = 0; i < shotResults.size(); i++) {
            std::string resultMark = shotResults[i] == 1 ? "O" : "X";
            cv::Scalar color = shotResults[i] == 1 ?
                cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);

            cv::putText(frame, resultMark, cv::Point(10 + i * 30, 60),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, color, 2);
        }
    }

    // 获取投篮统计信息
    void getStats(int& detected, int& made) {
        detected = shotsDetected;
        made = shotsMade;
    }
};

// 主函数
int main(int argc, char** argv) {
    std::string videoPath;

    // 如果有命令行参数，使用参数字符串路径
    if (argc == 2) {
        videoPath = argv[1];
    }
    // 否则要求用户输入视频路径
    else {
        std::cout << "请输入视频文件的完整路径: ";
        std::getline(std::cin, videoPath);
    }

    // 打开视频文件
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "无法打开视频文件!" << std::endl;
        return -1;
    }

    // 获取视频属性
    int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);

    std::cout << "视频分辨率: " << frameWidth << "x" << frameHeight << std::endl;
    std::cout << "帧率: " << fps << " FPS" << std::endl;

    // 创建篮球投篮分析器
    BasketballShotAnalyzer analyzer;

    // 设置篮筐位置（这里假设篮筐在画面中央偏上的位置）
    cv::Point hoopCenter(frameWidth * 0.53, frameHeight * 0.3);
    int hoopRadius = frameWidth / 25; // 大约是画面宽度的1/25
    analyzer.setupHoop(hoopCenter, hoopRadius);

    // 创建窗口
    cv::namedWindow("篮球投篮分析", cv::WINDOW_NORMAL);
    cv::resizeWindow("篮球投篮分析", 1280, 720);

    cv::Mat frame;
    bool isFirstFrame = true;

    std::cout << "开始处理视频..." << std::endl;
    std::cout << "按ESC键退出，按空格键暂停/继续" << std::endl;

    while (true) {
        // 读取视频帧
        cap >> frame;
        if (frame.empty()) {
            std::cout << "视频处理完成!" << std::endl;
            break;
        }

        // 如果是第一帧，显示设置信息
        if (isFirstFrame) {
            cv::Mat setupFrame = frame.clone();
            cv::putText(setupFrame, "Processing basketball video...",
                cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX,
                0.7, cv::Scalar(0, 255, 255), 2);

            cv::imshow("篮球投篮分析", setupFrame);
            cv::waitKey(1000); // 显示1秒

            isFirstFrame = false;
        }

        // 创建显示帧
        cv::Mat displayFrame = frame.clone();

        // 检测篮球
        cv::Point ballPosition = analyzer.detectBall(frame);

        // 更新过滤并检测投篮
        analyzer.updateTrajectory(ballPosition, displayFrame);

        // 绘制统计信息
        analyzer.drawStats(displayFrame);

        // 显示结果
        cv::imshow("篮球投篮分析", displayFrame);

        // 处理键盘输入
        int key = cv::waitKey(1);
        if (key == 27) break; // ESC键退出
        else if (key == 32) { // 空格键暂停/继续
            cv::waitKey(0);
        }

        // 获取当前统计结果
        int shotsDetected, shotsMade;
        analyzer.getStats(shotsDetected, shotsMade);

        // 如果已经检测到20次投篮，结束分析
        if (shotsDetected >= 20) {
            std::cout << "\n统计结果:" << std::endl;
            std::cout << "总投篮次数: 20" << std::endl;
            std::cout << "投进次数: " << shotsMade << std::endl;
            std::cout << "投篮命中率: " << (static_cast<double>(shotsMade) / 20.0) * 100 << "%" << std::endl;

            // 显示最终结果
            cv::Mat resultFrame = cv::Mat::zeros(480, 640, CV_8UC3);
            std::string finalResult = "15次投篮中有 " + std::to_string(shotsMade) + " 次投进";
            cv::putText(resultFrame, finalResult, cv::Point(50, 240),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);

            cv::imshow("篮球投篮分析结果", resultFrame);
            cv::waitKey(0);
            break;
        }
    
    }

    // 释放资源
    cap.release();
    cv::destroyAllWindows();

    return 0;

}