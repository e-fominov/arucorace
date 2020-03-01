#include <CLI11.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <deque>

static const int dictionary_id = cv::aruco::DICT_4X4_100;
static int capture_device_id = 1;
static std::set<int> good_markers = {0,1};
static int min_num_markers = 2;
static int marker_alive_time_ms = 300;
static int min_lap_time_ms = 3000;

void do_race()
{
    std::cout << "Starting aruco race" << std::endl;

    cv::Ptr<cv::aruco::DetectorParameters> params = cv::aruco::DetectorParameters::create();
    params->cornerRefinementMethod = cv::aruco::CORNER_REFINE_NONE;
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(dictionary_id);

    std::cout << "Opening video source with id=" << capture_device_id << std::endl;
    cv::VideoCapture cap(capture_device_id);
    cv::Mat mat;
    std::chrono::system_clock::time_point last_detection_time;
    int num_laps = 0;
    bool markers_detected = false;
    std::chrono::system_clock::time_point lap_begin_time = std::chrono::system_clock::now();

    std::deque<std::pair<int, int64_t>> lap_times;

    while (cap.read(mat))
    {
        std::chrono::system_clock::time_point now = std::chrono::system_clock::now();

        std::vector<int> markerIds;
        std::vector<std::vector<cv::Point2f>> markerCorners;
        cv::aruco::detectMarkers(mat, dictionary, markerCorners, markerIds, params);
        size_t num_good_markers = 0;
        for (size_t i = 0; i < markerIds.size();)
        {
            int id = markerIds[i];
            bool good_marker = good_markers.empty() || good_markers.count(id);
            if (good_marker)
                ++i;
            else
            {
                markerIds.erase(markerIds.begin() + i);
                markerCorners.erase(markerCorners.begin() + i);
            }
        }
        for (int i : markerIds)
            if (good_markers.empty() || good_markers.count(i))
                num_good_markers++;

        bool enough_markers_visible = num_good_markers >= min_num_markers;
        if (!enough_markers_visible && markers_detected)
        {
            auto elapsed = now - last_detection_time;
            if (elapsed > std::chrono::milliseconds(marker_alive_time_ms))
            {
                markers_detected = false;
                auto lap_time = now - lap_begin_time;
                int64_t lap_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(lap_time).count();
                if (lap_time_ms >= min_lap_time_ms)
                {
                    std::cout << "Lap " << num_laps << " time : " << lap_time_ms << " ms" << std::endl;
                    if (num_laps)
                        lap_times.emplace_back(num_laps, lap_time_ms);
                    if (lap_times.size() > 5)
                        lap_times.pop_front();
                    lap_begin_time = now;

                    ++num_laps;
                }

            }
        }
        else if (enough_markers_visible)
        {
            last_detection_time = now;
            markers_detected = true;
        }

        cv::aruco::drawDetectedMarkers(mat, markerCorners, markerIds);

        for (size_t i = 0; i < lap_times.size(); ++i)
        {
            std::ostringstream ss;
            int lap_id = lap_times[i].first;
            int64_t lap_time_ms = lap_times[i].second;

            ss << "Lap " << lap_id << " - " << std::fixed << std::setprecision(3) << lap_time_ms*0.001 << " sec";
            cv::putText(mat, ss.str(), cv::Point(10, 20 + i * 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(128, 255, 128), 1.5);
        }
        cv::resize(mat, mat, cv::Size(), 1000.0 / mat.cols, 1000.0 / mat.cols);
        cv::imshow("Race", mat);
        int key = cv::waitKey(1);
        if (key == 27)
            break;
    }
}
int main(int argc, char** argv) try {
    CLI::App app("Aruco race", "Arucorace");
    app.add_option("--dev", capture_device_id, "Capture device id", true);
    app.add_option("--keep-alive", marker_alive_time_ms, "Marker keep alive ms", true);
    app.add_option("--num-markers", min_num_markers, "Minimal number of visible markers", true);
    app.add_option("--min-lap-time", min_lap_time_ms, "Min lap time ms", true);
    app.callback(do_race);

    try {app.parse(argc, argv);}
    catch(CLI::Error& e)
    {return app.exit(e);}

    return EXIT_SUCCESS;
}
catch(std::exception& e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}
