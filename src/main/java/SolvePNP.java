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
    static MatOfPoint3f axisMat;
    static Mat rvect;
    static Mat tvect;
    static List<MatOfPoint> contours;
    static Point[] cornerPoints;

    public static void load(){
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        bottomObjPointsMat = new MatOfPoint3f();
        topObjPointsMat = new MatOfPoint3f();
        axisMat = new MatOfPoint3f();
        CameraCalibration calibration = new CameraCalibration();

        cameraMatrix = calibration.getCameraMatrix();
        distCoeffs = calibration.getDistCoeffs();

        rvect = new Mat();
        tvect = new Mat();

        frame = Imgcodecs.imread("RedLoading-048in-Down.jpg");
        Imgproc.resize(frame, frame, new Size(1080, 720));
    }


    public static void main(String[] args){
        load();
        setWorldCoordinates();

        contours = findContours(frame);
        cornerPoints = getPoints(contours.get(0), frame);
        getPose(cornerPoints);
        projectPoints(frame, getCenter(contours.get(0)));




    }

    public static Point getCenter(MatOfPoint contour){
        Rect rect = Imgproc.boundingRect(contour);
        return new Point(rect.x+rect.width/(double)2,rect.y+rect.height/(double)2 );
    }

    public static List<MatOfPoint> findContours(Mat frame){
        hsvThreshold(frame, new double[]{0, 255},
                new double[]{0, 255},
                new double[]{100, 255},
                frame
        );
        List<MatOfPoint> points = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(frame, points, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);
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
        Imgproc.cvtColor(frame, frame, Imgproc.COLOR_BGR2GRAY);
        TermCriteria aCriteria = new TermCriteria(TermCriteria.EPS |
                TermCriteria.MAX_ITER, 30,0.01);
        Imgproc.cornerSubPix(frame, corners, new Size(11,11), new Size(-1,-1), aCriteria);
        Mat _inliers = new Mat();
        Calib3d.solvePnPRansac(bottomObjPointsMat, corners, cameraMatrix, distCoeffs, rvect, tvect, true, 10, 0, 100.0,_inliers, Calib3d.SOLVEPNP_ITERATIVE);
        Imgproc.cvtColor(frame, frame, Imgproc.COLOR_GRAY2BGR);

    }

    public static void projectPoints(Mat frame, Point center){
        MatOfPoint2f bottomImagePoints = new MatOfPoint2f();
        MatOfPoint2f topImagePoints = new MatOfPoint2f();
        Calib3d.projectPoints(bottomObjPointsMat, rvect, tvect, cameraMatrix, distCoeffs, bottomImagePoints);
        Calib3d.projectPoints(topObjPointsMat, rvect, tvect, cameraMatrix, distCoeffs, topImagePoints);

        double bottomPeri = Imgproc.arcLength(new MatOfPoint2f(bottomImagePoints.toArray()), true);
        double topPeri = Imgproc.arcLength(new MatOfPoint2f(topImagePoints.toArray()), true);

        MatOfPoint2f bottomPolyOutput = new MatOfPoint2f();
        MatOfPoint2f topPolyOutput = new MatOfPoint2f();

        Imgproc.approxPolyDP(new MatOfPoint2f(bottomImagePoints.toArray()),bottomPolyOutput, 0.01 * bottomPeri, true );
        Imgproc.approxPolyDP(new MatOfPoint2f(topImagePoints.toArray()),topPolyOutput, 0.01 * topPeri, true );
        MatOfInt bottomPoints = new MatOfInt();
        Imgproc.convexHull(new MatOfPoint(bottomImagePoints.toArray()),bottomPoints );

        Imgproc.drawContours(frame, Collections.singletonList(new MatOfPoint(hull2Points(bottomPoints, new MatOfPoint(bottomImagePoints.toArray())).toArray())), -1, new Scalar(0, 0, 255), 2);

        for (int i = 0; i < bottomPolyOutput.rows(); i++) {
            Imgproc.line(frame, new Point(bottomPolyOutput.get(i, 0)), new Point(topPolyOutput.get(i, 0)), new Scalar(0, 255, 0), 2);
        }
        Mat frame2 = new Mat(frame.size(), frame.type());
        Imgproc.drawContours(frame2, Collections.singletonList(new MatOfPoint(bottomPolyOutput.toArray())), -1, new Scalar(0, 0, 255), 2);
        Imgproc.drawContours(frame2, List.of(contours.get(0)), -1, new Scalar(255, 0, 255), 2);



        MatOfInt hull = new MatOfInt();
        Imgproc.convexHull(contours.get(0),hull );
        Imgproc.drawContours(frame2, List.of(hull2Points(hull, contours.get(0))), 0, new Scalar(255, 0, 0), 1);


        HighGui.namedWindow("Frame",  HighGui.WINDOW_NORMAL);
        HighGui.imshow("Frame", frame);
        HighGui.waitKey();

    }

    public static MatOfPoint hull2Points(MatOfInt hull, MatOfPoint contour) {
        List<Integer> indexes = hull.toList();
        List<Point> points = new ArrayList<>();
        MatOfPoint point= new MatOfPoint();
        for(Integer index:indexes) {
            points.add(contour.toList().get(index));
        }
        point.fromList(points);
        return point;
    }

}
