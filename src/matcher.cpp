#include <fstream>
#include <iostream>
#include "database.hpp"
#include "instrumentation.hpp"
#include "matcher.hpp"
#include "sw.hpp"
#include "vocabulary.hpp"
#include <type_traits>
#include <opencv2/imgproc.hpp>

std::vector<std::pair<unsigned int, unsigned int>> thresholdScenes(const std::vector<double>& distances, double threshold) {
    std::vector<std::pair<unsigned int, unsigned int>> retval;

    unsigned int last = 0;
    for(unsigned int i = 0; i < distances.size(); i++) {
        if(distances[i] > threshold) {
            retval.emplace_back(last, i + 1);
            last = i + 1;
        }
    }
    retval.emplace_back(last, distances.size() + 1);

    return retval;
}

std::vector<std::pair<unsigned int, unsigned int>> hierarchicalScenes(const std::vector<double>& distances, int min_scene_length)
{
    std::vector<bool> excluded(distances.size(), 0);
    std::vector<std::pair<double, int>> sorted_distances(distances.size());

    std::transform(distances.begin(), distances.end(), sorted_distances.begin(), [index = 0](double val) mutable {
        return std::make_pair(val, index++);
    });

    std::vector<int> retval;
    std::sort(sorted_distances.begin(), sorted_distances.end(), [](auto l, auto r) { return l.first > r.first; });

    // iterate over distances from lagest to smallest
    for(auto current: sorted_distances)
    {
        if (excluded[current.second])
        {
            continue;
        }

        retval.push_back(current.second + 1);

        int low = std::max(current.second - min_scene_length, 0);
        int high = std::min(current.second + min_scene_length, (int)distances.size());

        // exclude neighbors of current
        for (int i = low; i < high; i++)
        {
            excluded[i] = 1;
        }
    }

    retval.push_back(distances.size() + 1);

    // return cutoffs from left to right
    std::sort(retval.begin(), retval.end());

    std::vector<std::pair<unsigned int, unsigned int>> v;
    std::transform(retval.begin(), retval.end(), std::back_inserter(v), [last = 0](auto I) mutable { auto r = std::make_pair(last, I); last = I; return r; });

    return v;
}

Vocab<Frame> constructFrameVocabulary(const FileDatabase &database, unsigned int K, unsigned int speedinator)
{
    cv::Mat descriptors;

    for (auto video : database.listVideos())
    {
        auto v = database.loadVideo(video);
        auto frames = v->frames();
        while(auto i = frames->read())
            descriptors.push_back(i->descriptors);
    }
    cv::UMat copy;
    descriptors.copyTo(copy);

    return Vocab<Frame>(constructVocabulary(copy, K));
}

Vocab<SerializableScene> constructSceneVocabulary(const FileDatabase &database, unsigned int K, unsigned int speedinator)
{
    auto vocab = loadVocabulary<Frame>(database);
    if (!vocab)
    {
        throw std::runtime_error("trying to construct frame vocab but sift vocab is empty");
    }
    auto d = BOWExtractor(vocab->descriptors());

    cv::Mat descriptors;

    for (auto video : database.listVideos())
    {
        auto v = database.loadVideo(video);
        auto frames = v->frames();
        while(auto i = frames->read())
            descriptors.push_back(getFrameDescriptor(*i, d));
    }

    cv::UMat copy;
    descriptors.copyTo(copy);

    std::cout << "\u001b[37m" <<"Constructing Scene Vocabulary. Processing " << descriptors.rows << " frame descriptors" << std::endl;

    return Vocab<SerializableScene>(constructVocabulary(copy, K));
}

void overflow(std::vector<cv::Mat> &descriptor_levels, unsigned int K, unsigned int N, unsigned int level)
{
    // FIXME: better solution to not having enough points, maybe throw away?
    //while(descriptor_levels[level].rows < K){
    //    descriptor_levels[level].push_back();
    //}

    // std::cout << "Enter overflow at level " << level << ", shape is: ";
    for (auto &a : descriptor_levels)
    {
        // std::cout << a.rows << ">";
    }
    // std::cout << std::endl;

    // throw away leftovers
    if (descriptor_levels[level].rows < K)
    {
        descriptor_levels[level] = cv::Mat();
        // std::cout << "Early return" << std::endl;
        return;
    }

    // make next level if none
    if (level == descriptor_levels.size() - 1)
    {
        descriptor_levels.push_back(cv::Mat());
    }
    cv::UMat copy;
    descriptor_levels[level].copyTo(copy);

    descriptor_levels[level + 1].push_back(constructVocabulary(copy, K));
    descriptor_levels[level] = cv::Mat();

    // if next level is too large, reduce with kmeans
    if (descriptor_levels[level + 1].rows >= N)
    {
        overflow(descriptor_levels, K, N, level + 1);
    }

    // std::cout << "Exit overflow at level " << level << ", shape is: ";
    for (auto &a : descriptor_levels)
    {
        // std::cout << a.rows << ">";
    }
    // std::cout << std::endl;
}
cv::Mat constructDemoVocabularyHierarchical(const std::vector<cv::Point2f>& points, unsigned int K, unsigned int N) {
    if (K > N)
    {
        throw std::runtime_error("constructFrameVocabularyHierarchical: error, K > N");
    }

    //cv::Mat descriptors;

    std::vector<cv::Mat> descriptor_levels(1);
    unsigned int total_rows = 0;

    for (auto point : points)
    {
        // counter++;
        // if (counter % speedinator != 0) {  continue;  }
        descriptor_levels[0].push_back(point);
        // limit largest kmeans run
        if (descriptor_levels[0].rows >= N)
        {
            total_rows += descriptor_levels[0].rows;
            std::cout << "Constructing Vocabulary. Processing " << total_rows << " points\r";
            std::cout.flush();
            overflow(descriptor_levels, K, N, 0);
        }
    }

    // flush all remaining features to lowest level
    for (int i = 0; i < descriptor_levels.size(); i++)
    {
        // if K == 2000 and at lowest level this is the vocabulary
        if (descriptor_levels[i].rows != K || i != descriptor_levels.size() - 1)
        {
            total_rows += descriptor_levels[i].rows;
            std::cout << "Constructing Vocabulary. Processing " << total_rows << " points\r";
            std::cout.flush();
            overflow(descriptor_levels, K, N, i);
        }
    }
    std::cout << std::endl;
    // can return empty matrix if data has fewer than K descriptors
    return descriptor_levels.back();
}
// N is maximum size on which to perform clustering
Vocab<Frame> constructFrameVocabularyHierarchical(const FileDatabase &database, unsigned int K, unsigned int N, unsigned int speedinator)
{
    if (K > N)
    {
        throw std::runtime_error("constructFrameVocabularyHierarchical: error, K > N");
    }

    //cv::Mat descriptors;

    std::vector<cv::Mat> descriptor_levels(1);
    unsigned int total_rows = 0;

    // collect features and reduce with kmeans as needed
    for (auto video : database.listVideos())
    {
        auto v = database.loadVideo(video);
        auto frames = v->frames();
        // unsigned int counter = 0;
        while(auto i = frames->read())
        {
            // counter++;
            // if (counter % speedinator != 0) {  continue;  }
            descriptor_levels[0].push_back(i->descriptors);
            // limit largest kmeans run
            if (descriptor_levels[0].rows >= N)
            {
                total_rows += descriptor_levels[0].rows;
                std::cout << "\u001b[37m" <<"Constructing Frame Vocabulary. Processing " << total_rows << " SIFT features\r";
                std::cout.flush();
                overflow(descriptor_levels, K, N, 0);
            }

            frames ->skip(speedinator);
        }
    }

    // flush all remaining features to lowest level
    for (int i = 0; i < descriptor_levels.size(); i++)
    {
        // if K == 2000 and at lowest level this is the vocabulary
        if (descriptor_levels[i].rows != K || i != descriptor_levels.size() - 1)
        {
            total_rows += descriptor_levels[i].rows;
            std::cout << "\u001b[37m" <<"Constructing Frame Vocabulary. Processing " << total_rows << " SIFT features\r";
            std::cout.flush();
            overflow(descriptor_levels, K, N, i);
        }
    }
    std::cout << std::endl;
    // can return empty matrix if data has fewer than K descriptors
    return Vocab<Frame>(descriptor_levels.back());
}

// N is maximum size on which to perform clustering
Vocab<SerializableScene> constructSceneVocabularyHierarchical(const FileDatabase &database, unsigned int K, unsigned int N, unsigned int speedinator)
{
    if (K > N)
    {
        throw std::runtime_error("constructFrameVocabularyHierarchical: error, K > N");
    }

    auto vocab = loadVocabulary<Frame>(database);
    if (!vocab)
    {
        throw std::runtime_error("trying to construct frame vocab but sift vocab is empty");
    }
    auto d = BOWExtractor(vocab->descriptors());

    //cv::Mat descriptors;

    std::vector<cv::Mat> descriptor_levels(1);

    // collect features, reduce if needed
    for (auto video : database.listVideos())
    {
        auto v = database.loadVideo(video);
        auto frames = v->frames();
        while(auto i = frames->read()){
            descriptor_levels.push_back(loadFrameDescriptor(*i, d));
			if(descriptor_levels[0].rows >= N){
				overflow(descriptor_levels, K, N, 0);
			}

            frames ->skip(speedinator);
		}
    }

    // flush all remaining features to lowest level
    for (int i = 0; i < descriptor_levels.size(); i++)
    {
        // if K == 2000 and at lowest level this is the vocabulary
        if (descriptor_levels[i].rows != K || i != descriptor_levels.size() - 1)
        {
            overflow(descriptor_levels, K, N, i);
        }
    }

    // can return empty matrix if data has fewer than K descriptors
    return Vocab<SerializableScene>(descriptor_levels.back());
}
/*
double boneheadedSimilarity(IVideo &v1, IVideo &v2, std::function<double(Frame, Frame)> comparator, SimilarityReporter reporter)
{
    auto frames1 = v1.frames();
    auto frames2 = v2.frames();

    double total = 0;

    auto len = std::min(v1.frameCount(), v2.frameCount());

    for (decltype(len) i = 0; i < len; i++)
    {
        auto f1 = frames1->read();
        auto f2 = frames2->read();
        auto t = comparator(f1, f2);
        if (reporter)
            reporter(FrameSimilarityInfo{t, i, i,
                                         v1, v2});

        total += (t != -1) ? t : 0;
    }

    return total / len;
}*/

template<typename Reader>
std::vector<MatchInfo> internal_findMatch(Reader&& reader, const FileDatabase &db)
{
    typedef typename decltype(reader.read())::value_type value_type;
    auto intcomp = [](auto f1, auto f2) { return cosineSimilarity(f1, f2) > 0.8 ? 3 : -3; };

    std::vector<MatchInfo> match;

    // load query information into array
    std::vector<cv::Mat> targetDescriptors;
    std::vector<std::pair<unsigned int, unsigned int>> query_scenes;

    if constexpr(std::is_same_v<value_type, SerializableScene>) {
        while(auto scene = reader.read()) {
            targetDescriptors.push_back(scene->frameBag);
            query_scenes.emplace_back(scene->startIdx, scene->endIdx);
        }
    }
    else if constexpr(std::is_same_v<value_type, Frame>)
        while(auto frame = reader.read()) targetDescriptors.push_back(frame->frameDescriptor);

    // calculate alignments for each video in database
    for (const std::string& v2 : db.listVideos())
    {
        std::cout << "Calculating match for " << v2 << std::endl;
        std::vector<cv::Mat> knownDescriptors;
        std::vector<std::pair<unsigned int, unsigned int>> db_scenes;
        auto v = db.loadVideo(v2);
        double frameRate = v->loadMetadata().frameRate;

        if constexpr(std::is_same_v<value_type, SerializableScene>) {
            auto scenes = v->getScenes();
            while(auto scene = scenes->read()) {
                knownDescriptors.push_back(scene->frameBag);
                db_scenes.emplace_back(scene->startIdx, scene->endIdx);
            }
        } else if constexpr(std::is_same_v<value_type, Frame>) {
            auto frames = v->frameBags();
            while(auto frame = frames->read()) knownDescriptors.push_back(*frame);
        }

        auto &&alignments = calculateAlignment(knownDescriptors, targetDescriptors, intcomp, 3, 2);
        // std::cout << targetScenes.size() << " " << knownScenes.size() << std::endl;
        if (alignments.size() > 0)
        {
            std::cout << "Score: " << alignments[0].score << std::endl;
            for (auto a: alignments)
            {
                if constexpr(std::is_same_v<value_type, SerializableScene>) {
                    double tot = 0;
                    int num = 0;
                    for(int i = a.startUnknown; i < a.endUnknown; i++){
                            double best = -1;

                        for(int j = a.startKnown; j < a.endKnown; j++){
                            if(double t; (t = cosineSimilarity(targetDescriptors[i], knownDescriptors[j])) > best){
                                best = t;
                            }
                        }

                        tot += best;
                        num++;
                    }

                    double avg = tot / num;

                    match.push_back(MatchInfo{v->name,
                        static_cast<double>(a.score),
                        frameRate,
                        query_scenes[a.startUnknown].first,
                        query_scenes[a.endUnknown - 1].second,
                        db_scenes[a.startKnown].first,
                        db_scenes[a.endKnown - 1].second,
                        avg});

                } else if constexpr(std::is_same_v<value_type, Frame>)
                    match.push_back(MatchInfo{v->name,
                        static_cast<double>(a.score),
                        frameRate,
                        a.startUnknown,
                        a.endUnknown,
                        a.startKnown,
                        a.endKnown});
            }
        } else {
            match.push_back(MatchInfo{v->name, 0, frameRate, 0, 0, 0, 0});
        }
    }

    std::sort(match.begin(), match.end(), [](const MatchInfo& a, const MatchInfo& b){
        return a.confidence > b.confidence;
    });

    return match;
}

std::vector<MatchInfo> findMatch(QueryVideo& video, const FileDatabase &db) {
    return internal_findMatch(*video.getScenes(), db);
}

std::vector<MatchInfo> findMatch(QueryVideo&& video, const FileDatabase &db) {
    return internal_findMatch(*video.getScenes(), db);
}

std::vector<MatchInfo> findMatch(std::unique_ptr<ICursor<Frame>> frames, const FileDatabase &db) {
    return internal_findMatch(*frames, db);
}

std::vector<MatchInfo> findMatch(std::unique_ptr<ICursor<cv::Mat>> frames, const FileDatabase &db) {
    return internal_findMatch(*frames, db);
}

double BOWComparator::operator()(Frame &f1, Frame &f2)
{
    return frameSimilarity(f1, f2, [this](Frame &f) { return loadFrameDescriptor(f, extractor); });
}

double BOWComparator::operator()(Frame &f1, Frame &f2) const
{
    return frameSimilarity(f1, f2, [this](Frame &f) {
        BOWExtractor copy{extractor};
        return loadFrameDescriptor(f, copy);
    });
}

double ColorComparator2D::operator()(const Frame &f1, const Frame &f2) const
{
    return operator()(f1.colorHistogram, f2.colorHistogram);
}
double ColorComparator2D::operator()(const cv::Mat &f1, const cv::Mat &f2) const
{
    if (f1.size() != f2.size())
    {
        throw std::runtime_error("colorHistograms not matching");
    }

    auto subbed = f1 - f2;
    auto val = cv::sum(subbed)[0];
    return std::abs(val);
}

double ColorIntersectComparator::operator()(const Frame &f1, const Frame &f2) const
{
    return operator()(f1.colorHistogram, f2.colorHistogram);
}
double ColorIntersectComparator::operator()(const cv::Mat &f1, const cv::Mat &f2) const
{
    if (f1.size() != f2.size())
    {
        throw std::runtime_error("colorHistograms not matching");
    }

    return 1 - cv::compareHist(f1, f2, cv::HISTCMP_INTERSECT);
}
