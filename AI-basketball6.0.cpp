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

    // �Ľ��Ĳ���
    int minBallRadius;
    int maxBallRadius;
    double ballDetectionConfidenceThreshold;
    int trajectoryBufferSize;
    int minTrajectoryPointsForShot;
    double minMovementThreshold;
    int maxMissingFrames;
    int missingFrameCount;
    cv::Point lastValidBallPos;

    // ����������
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
        // ��������������
        backgroundSubtractor = cv::createBackgroundSubtractorMOG2(500, 16, true);
    }

    void setupHoop(const cv::Point& center, int radius) {
        hoopCenter = center;
        hoopRadius = radius;
        std::cout << "����������� - ����: (" << center.x << ", " << center.y
            << "), �뾶: " << radius << std::endl;
    }

    // �Ľ��������㷨
    cv::Point detectBall(const cv::Mat& frame) {
        cv::Mat processFrame = frame.clone();

        // ����1��������ɫ�ļ�⣨��ɫ��
        cv::Point colorBasedBall = detectBallByColor(processFrame);

        // ����2�����ڱ��������ļ��
        cv::Point motionBasedBall = detectBallByMotion(processFrame);

        // ����3������Բ�μ��
        cv::Point shapeBasedBall = detectBallByShape(processFrame);

        // �ں϶��ּ����
        cv::Point finalBall = fuseBallDetections(colorBasedBall, motionBasedBall, shapeBasedBall);

        // ʹ�ü򵥵�λ����֤��ƽ������
        return smoothBallPosition(finalBall);
    }

    // ���¹��˲����Ͷ��
    void updateTrajectory(const cv::Point& ballPos, cv::Mat& displayFrame) {
        // ����ҵ�����
        if (ballPos.x >= 0 && ballPos.y >= 0) {
            missingFrameCount = 0;

            // ����ƶ��Ƿ��㹻�󣨱���������
            if (lastValidBallPos.x >= 0) {
                double movement = cv::norm(ballPos - lastValidBallPos);
                if (movement < minMovementThreshold && !processingShot) {
                    return; // �ƶ�̫С������������
                }
            }

            // ����λ����ʷ
            previousBallPositions.push_back(ballPos);
            if (static_cast<int>(previousBallPositions.size()) > trajectoryBufferSize) {
                previousBallPositions.erase(previousBallPositions.begin());
            }

            lastValidBallPos = ballPos;

            // ��ʾ���λ��
            cv::circle(displayFrame, ballPos, 8, cv::Scalar(0, 255, 255), -1);

            // ����µ�Ͷ����ʼ
            if (!processingShot && previousBallPositions.size() >= 5) {
                if (detectShotStart()) {
                    processingShot = true;
                    ballTrajectory.clear();
                    std::cout << "��⵽�µ�Ͷ����" << std::endl;
                }
            }

            // ������ڸ���Ͷ������¼����
            if (processingShot) {
                ballTrajectory.push_back(ballPos);
                checkShotCompletion(displayFrame);
            }
        }
        else {
            missingFrameCount++;

            // ������ڴ���Ͷ���Ҷ�ʧ֡����̫�࣬��������
            if (processingShot && missingFrameCount <= maxMissingFrames) {
                if (!ballTrajectory.empty()) {
                    // ʹ�����һ����֪λ��
                    ballTrajectory.push_back(ballTrajectory.back());
                }
                checkShotCompletion(displayFrame);
            }
            else if (missingFrameCount > maxMissingFrames && processingShot) {
                // ��ʧ̫��֡��������ǰͶ������
                processingShot = false;
                ballTrajectory.clear();
                std::cout << "���ٶ�ʧ������Ͷ�����" << std::endl;
            }
        }
    }

private:
    // ������ɫ������
    cv::Point detectBallByColor(const cv::Mat& frame) {
        cv::Mat hsvFrame;
        cv::cvtColor(frame, hsvFrame, cv::COLOR_BGR2HSV);

        // ������ɫ��Χ�������ڲ�ͬ��������
        cv::Scalar lowerOrange1(0, 120, 70);
        cv::Scalar upperOrange1(10, 255, 255);
        cv::Scalar lowerOrange2(170, 120, 70);
        cv::Scalar upperOrange2(180, 255, 255);

        cv::Mat mask1, mask2, mask;
        cv::inRange(hsvFrame, lowerOrange1, upperOrange1, mask1);
        cv::inRange(hsvFrame, lowerOrange2, upperOrange2, mask2);
        cv::bitwise_or(mask1, mask2, mask);

        // ��̬ѧ����
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);

        return findBestCircularContour(mask);
    }

    // �����˶�������
    cv::Point detectBallByMotion(const cv::Mat& frame) {
        if (!useBackgroundSubtraction) return cv::Point(-1, -1);

        cv::Mat fgMask;
        backgroundSubtractor->apply(frame, fgMask);

        // ȥ������
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::morphologyEx(fgMask, fgMask, cv::MORPH_OPEN, kernel);

        return findBestCircularContour(fgMask);
    }

    // ������״������
    cv::Point detectBallByShape(const cv::Mat& frame) {
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // ��˹ģ��
        cv::GaussianBlur(gray, gray, cv::Size(9, 9), 2, 2);

        // ����Բ���
        std::vector<cv::Vec3f> circles;
        cv::HoughCircles(gray, circles, cv::HOUGH_GRADIENT, 1,
            static_cast<double>(gray.rows) / 8, 100, 30, minBallRadius, maxBallRadius);

        if (!circles.empty()) {
            // ѡ������ܵ�Բ
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

    // ���������ҵ���õ�Բ������
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

            // �ۺ����֣�Բ�ζ� + ��С���� + λ��������
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

    // �ں϶��ּ����
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

        // ���ֻ��һ�������
        if (validDetections.size() == 1) {
            return validDetections[0];
        }

        // ����ƽ��λ�û�ѡ����ӽ���һ��λ�õ�
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

        // ����ƽ��λ��
        int avgX = 0, avgY = 0;
        for (const auto& detection : validDetections) {
            avgX += detection.x;
            avgY += detection.y;
        }
        return cv::Point(avgX / static_cast<int>(validDetections.size()),
            avgY / static_cast<int>(validDetections.size()));
    }

    // ƽ�����λ��
    cv::Point smoothBallPosition(const cv::Point& detectedBall) {
        if (detectedBall.x < 0) return detectedBall;

        // �򵥵�ƽ�����������һ��λ�þ���̫�󣬽��в�ֵ
        if (lastValidBallPos.x >= 0) {
            double distance = cv::norm(detectedBall - lastValidBallPos);
            if (distance > 80 && !processingShot) {
                // ����̫�󣬿�������죬������Чλ��
                return cv::Point(-1, -1);
            }
        }

        return detectedBall;
    }

    // ���Ͷ����ʼ
    bool detectShotStart() {
        if (previousBallPositions.size() < 5) return false;

        // ������Ƿ��������˶�
        int recentFrames = 5;
        int upwardMovement = 0;

        for (int i = previousBallPositions.size() - recentFrames;
            i < static_cast<int>(previousBallPositions.size()) - 1; i++) {
            if (previousBallPositions[i].y > previousBallPositions[i + 1].y) {
                upwardMovement++;
            }
        }

        // ����󲿷�֡����ʾ�����˶�����Ϊ��Ͷ����ʼ
        return upwardMovement >= 3;
    }

    // ���Ͷ�����
    void checkShotCompletion(cv::Mat& displayFrame) {
        if (static_cast<int>(ballTrajectory.size()) < minTrajectoryPointsForShot) return;

        // �������Ƿ�ͨ������
        bool passedThroughHoop = false;
        bool isDescending = false;
        int pointsNearHoop = 0;

        // ����������������
        int recentPoints = std::min(8, static_cast<int>(ballTrajectory.size()));
        if (recentPoints >= 3) {
            int startIdx = ballTrajectory.size() - recentPoints;
            isDescending = ballTrajectory[startIdx].y < ballTrajectory[ballTrajectory.size() - 1].y;
        }

        // �����ж��ٹ��˵�ӽ�����
        for (const auto& point : ballTrajectory) {
            double distance = cv::norm(point - hoopCenter);
            if (distance < hoopRadius * 1.5) {
                pointsNearHoop++;
            }
        }

        // �ж��Ƿ�ͨ������
        if (pointsNearHoop > 2) {
            passedThroughHoop = true;
        }

        // ������Ѿ�ͨ�������ҿ�ʼ�½������Ͷ��
        if (passedThroughHoop && isDescending) {
            shotsDetected++;

            // �ж��Ƿ�Ͷ��
            bool isMade = determineIfShotMade();
            if (isMade) {
                shotsMade++;
                shotResults.push_back(1);
                std::cout << "Ͷ�� #" << shotsDetected << ": Ͷ���ˣ�" << std::endl;
            }
            else {
                shotResults.push_back(0);
                std::cout << "Ͷ�� #" << shotsDetected << ": δͶ����" << std::endl;
            }

            // ����״̬��׼����һ��Ͷ��
            processingShot = false;
            ballTrajectory.clear();

            // �ڻ�������ʾͶ�����
            std::string resultText = "Ͷ�� #" + std::to_string(shotsDetected) +
                ": " + (isMade ? "����" : "δ��");
            cv::putText(displayFrame, resultText, cv::Point(50, 50 + shotsDetected * 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.7,
                isMade ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), 2);
        }
    }

    // ͨ�����������ж����Ƿ��������
    bool determineIfShotMade() {
        // �����ж��ٵ�����������
        int pointsInHoop = 0;
        for (const auto& point : ballTrajectory) {
            double distance = cv::norm(point - hoopCenter);
            if (distance < hoopRadius * 0.8) {
                pointsInHoop++;
            }
        }

        // ������ĩ���Ƿ��������·�
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
    // ͳ����Ϣ��ʾ����
    void drawStats(cv::Mat& frame) {
        // ����������
        double shootingPercentage = shotsDetected > 0 ?
            static_cast<double>(shotsMade) / shotsDetected * 100.0 : 0.0;

        // ��ʾ����
        cv::circle(frame, hoopCenter, hoopRadius, cv::Scalar(0, 165, 255), 2);

        // ��ʾͳ����Ϣ
        std::string statsText = "��⵽��Ͷ��: " + std::to_string(shotsDetected) +
            ", Ͷ��: " + std::to_string(shotsMade) +
            " (" + std::to_string(static_cast<int>(shootingPercentage)) + "%)";

        cv::putText(frame, statsText, cv::Point(10, 30),
            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);

        // ��ʾÿ��Ͷ���Ľ��
        for (size_t i = 0; i < shotResults.size(); i++) {
            std::string resultMark = shotResults[i] == 1 ? "O" : "X";
            cv::Scalar color = shotResults[i] == 1 ?
                cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);

            cv::putText(frame, resultMark, cv::Point(10 + i * 30, 60),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, color, 2);
        }
    }

    // ��ȡͶ��ͳ����Ϣ
    void getStats(int& detected, int& made) {
        detected = shotsDetected;
        made = shotsMade;
    }
};

// ������
int main(int argc, char** argv) {
    std::string videoPath;

    // ����������в�����ʹ�ò����ַ���·��
    if (argc == 2) {
        videoPath = argv[1];
    }
    // ����Ҫ���û�������Ƶ·��
    else {
        std::cout << "��������Ƶ�ļ�������·��: ";
        std::getline(std::cin, videoPath);
    }

    // ����Ƶ�ļ�
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "�޷�����Ƶ�ļ�!" << std::endl;
        return -1;
    }

    // ��ȡ��Ƶ����
    int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);

    std::cout << "��Ƶ�ֱ���: " << frameWidth << "x" << frameHeight << std::endl;
    std::cout << "֡��: " << fps << " FPS" << std::endl;

    // ��������Ͷ��������
    BasketballShotAnalyzer analyzer;

    // ��������λ�ã�������������ڻ�������ƫ�ϵ�λ�ã�
    cv::Point hoopCenter(frameWidth * 0.53, frameHeight * 0.3);
    int hoopRadius = frameWidth / 25; // ��Լ�ǻ����ȵ�1/25
    analyzer.setupHoop(hoopCenter, hoopRadius);

    // ��������
    cv::namedWindow("����Ͷ������", cv::WINDOW_NORMAL);
    cv::resizeWindow("����Ͷ������", 1280, 720);

    cv::Mat frame;
    bool isFirstFrame = true;

    std::cout << "��ʼ������Ƶ..." << std::endl;
    std::cout << "��ESC���˳������ո����ͣ/����" << std::endl;

    while (true) {
        // ��ȡ��Ƶ֡
        cap >> frame;
        if (frame.empty()) {
            std::cout << "��Ƶ�������!" << std::endl;
            break;
        }

        // ����ǵ�һ֡����ʾ������Ϣ
        if (isFirstFrame) {
            cv::Mat setupFrame = frame.clone();
            cv::putText(setupFrame, "Processing basketball video...",
                cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX,
                0.7, cv::Scalar(0, 255, 255), 2);

            cv::imshow("����Ͷ������", setupFrame);
            cv::waitKey(1000); // ��ʾ1��

            isFirstFrame = false;
        }

        // ������ʾ֡
        cv::Mat displayFrame = frame.clone();

        // �������
        cv::Point ballPosition = analyzer.detectBall(frame);

        // ���¹��˲����Ͷ��
        analyzer.updateTrajectory(ballPosition, displayFrame);

        // ����ͳ����Ϣ
        analyzer.drawStats(displayFrame);

        // ��ʾ���
        cv::imshow("����Ͷ������", displayFrame);

        // �����������
        int key = cv::waitKey(1);
        if (key == 27) break; // ESC���˳�
        else if (key == 32) { // �ո����ͣ/����
            cv::waitKey(0);
        }

        // ��ȡ��ǰͳ�ƽ��
        int shotsDetected, shotsMade;
        analyzer.getStats(shotsDetected, shotsMade);

        // ����Ѿ���⵽20��Ͷ������������
        if (shotsDetected >= 20) {
            std::cout << "\nͳ�ƽ��:" << std::endl;
            std::cout << "��Ͷ������: 20" << std::endl;
            std::cout << "Ͷ������: " << shotsMade << std::endl;
            std::cout << "Ͷ��������: " << (static_cast<double>(shotsMade) / 20.0) * 100 << "%" << std::endl;

            // ��ʾ���ս��
            cv::Mat resultFrame = cv::Mat::zeros(480, 640, CV_8UC3);
            std::string finalResult = "15��Ͷ������ " + std::to_string(shotsMade) + " ��Ͷ��";
            cv::putText(resultFrame, finalResult, cv::Point(50, 240),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);

            cv::imshow("����Ͷ���������", resultFrame);
            cv::waitKey(0);
            break;
        }
    
    }

    // �ͷ���Դ
    cap.release();
    cv::destroyAllWindows();

    return 0;

}