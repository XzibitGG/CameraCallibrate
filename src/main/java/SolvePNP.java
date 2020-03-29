import org.opencv.calib3d.Calib3d;
import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class SolvePNP {

    static Mat cameraMatrix;
    static MatOfDouble distCoeffs;
    static Mat frame;
    static MatOfPoint3f bottomObjPointsMat;
    static MatOfPoint3f topObjPointsMat;
    static Mat rvect;
    static Mat tvect;

    public static void load(){
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        bottomObjPointsMat = new MatOfPoint3f();
        topObjPointsMat = new MatOfPoint3f();
        CameraCalibration calibration = new CameraCalibration();

        cameraMatrix = calibration.getCameraMatrix();
        distCoeffs = calibration.getDistCoeffs();

        rvect = new Mat();
        tvect = new Mat();

        frame = Imgcodecs.imread("LoadingBay.jpg");
    }


    public static void main(String[] args){
        load();
        setWorldCoordinates();

        List<MatOfPoint> contours = findContours(frame);
        Point[] cornerPoints = getPoints(contours.get(0), frame);
        drawCorners(cornerPoints, frame);

        getPose(cornerPoints);
       projectPoints(frame, getCenter(contours.get(0)));

        HighGui.imshow("Frame", frame);
        HighGui.waitKey();


    }

    public static Point getCenter(MatOfPoint contour){
        Rect rect = Imgproc.boundingRect(contour);
        return new Point(rect.x+rect.width/(double)2,rect.y+rect.height/(double)2 );
    }

    public static void drawCorners(Point[] corners, Mat frame){
        for(Point point : corners){
            Imgproc.circle(frame, point, 4,  new Scalar(0, 255, 0) );
        }
    }

    public static List<MatOfPoint> findContours(Mat frame){
        hsvThreshold(frame, new double[]{0, 255},
                new double[]{0, 255},
                new double[]{100, 255},
                frame
        );
        List<MatOfPoint> points = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(frame, points, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);
        //Sort by largest
        points.sort((o1, o2) -> {
            double area1 = Imgproc.contourArea(o1);
            double area2 = Imgproc.contourArea(o2);
            return Double.compare(area1, area2);
        });
        Collections.reverse(points);
        Imgproc.cvtColor(frame, frame, Imgproc.COLOR_GRAY2BGR);
        return points;
    }

    public static Point[] getPoints(MatOfPoint contour, Mat frame){
        MatOfPoint2f contourMat = new MatOfPoint2f();
        contourMat.fromArray(contour.toArray());
        RotatedRect rect = Imgproc.minAreaRect(contourMat);
        Point[] cornersPoints = new Point[4];
        rect.points(cornersPoints);

        for(int i=0; i<4; ++i){
            Imgproc.line(frame, cornersPoints[i], cornersPoints[(i+1)%4], new Scalar(0,255,0));
        }
        return cornersPoints;
    }

    private static void hsvThreshold(Mat input, double[] hue, double[] sat, double[] val, Mat out) {
        Imgproc.cvtColor(input, out, Imgproc.COLOR_BGR2HSV);
        Core.inRange(out, new Scalar(hue[0], sat[0], val[0]),
                new Scalar(hue[1], sat[1], val[1]), out);
    }

    private static void setWorldCoordinates() {
        List<Point3> bottomCorners = List.of(
                new Point3(-7 / 2.0, 11 / 2.0, 0.0),
                new Point3(-7 / 2.0, -11 / 2.0, 0.0),
                new Point3(7 / 2.0, -11 / 2.0, 0.0),
                new Point3(7 / 2.0, 11 / 2.0, 0.0)
        );
        bottomObjPointsMat.fromList(bottomCorners);

        List<Point3> topCorners = List.of(
                new Point3(-7 / 2.0, 11 / 2.0, -3.0),
                new Point3(-7 / 2.0, -11 / 2.0, -3.0),
                new Point3(7 / 2.0, -11 / 2.0, -3.0),
                new Point3(7 / 2.0, 11 / 2.0, -3.0)
        );
        topObjPointsMat.fromList(topCorners);
    }

    public static  void getPose(Point[] imgCorners){
        MatOfPoint2f corners = new MatOfPoint2f();
        corners.fromArray(imgCorners);
        Calib3d.solvePnP(bottomObjPointsMat, corners, cameraMatrix, distCoeffs, rvect, tvect);

        //System.out.println(rvect.dump() + " " + tvect.dump());
    }

    public static void projectPoints(Mat frame, Point center){
        MatOfPoint2f bottomImagePoints = new MatOfPoint2f();
        MatOfPoint2f topImagePoints = new MatOfPoint2f();
        Calib3d.projectPoints(bottomObjPointsMat, rvect, tvect, cameraMatrix, distCoeffs, bottomImagePoints);
        Calib3d.projectPoints(topObjPointsMat, rvect, tvect, cameraMatrix, distCoeffs, topImagePoints);


        Imgproc.drawContours(frame, Collections.singletonList(new MatOfPoint(bottomImagePoints.toArray())), -1, new Scalar(0, 0, 255), 2);

        for (int i = 0; i < bottomImagePoints.rows(); i++) {
            Imgproc.line(frame, new Point(bottomImagePoints.get(i, 0)), new Point(topImagePoints.get(i, 0)), new Scalar(0, 255, 0), 2);
        }

        Imgproc.drawContours(frame, Collections.singletonList(new MatOfPoint(topImagePoints.toArray())), -1, new Scalar(255, 0, 255), 2);

    }


}
