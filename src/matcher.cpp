#include <fstream>
#include <iostream>
#include "database.hpp"
#include "instrumentation.hpp"
#include "matcher.hpp"
#include "sw.hpp"
#include "vocabulary.hpp"
#include <boost/range/adaptors.hpp>
#include <boost/range/algorithm_ext/push_back.hpp>

std::vector<float> get_distances(FileLoader fl, std::string video_path)
{
    cv::Mat current, next;
    std::vector<float> retval;

    auto rv = fl.readFrameColorHistogram(video_path, 0);
    if (rv)
    {
        current = *rv;
    }
    else
    {
        return {};
    }

    int index = 0;
    ColorComparator cc;

    while (1)
    {
        index++;

        rv = fl.readFrameColorHistogram(video_path, index);
        if (rv)
        {
            next = *rv;
            retval.push_back(cc(current, next));
            current = next;
        }
        else
        {
            return retval;
        }
    }
}

std::vector<std::pair<int, int>> hierarchicalScenes(std::vector<float> distances, int min_scene_length)
{
    std::vector<bool> excluded(distances.size(), 0);
    std::vector<std::pair<float, int>> sorted_distances;

    for (int i = 0; i < distances.size(); i++)
    {
        sorted_distances.push_back({distances[i], i});
    }

    std::vector<int> retval;
    std::sort(sorted_distances.begin(), sorted_distances.end(), [](auto l, auto r) { return l.first > r.first; });

    int index = -1;
    std::pair<float, int> current;

    // iterate over distances from lagest to smallest
    while (index < (int)distances.size() - 1)
    {
        index++;
        current = sorted_distances[index];
        if (excluded[current.second])
        {
            continue;
        }

        retval.push_back(current.second);

        int low = std::max(current.second - min_scene_length, 0);
        int high = std::min(current.second + min_scene_length, (int)distances.size());

        // exclude neighbors of current
        for (int i = low; i < high; i++)
        {
            excluded[i] = 1;
        }
    }

    // return cutoffs from left to right
    std::sort(retval.begin(), retval.end());

    std::vector<std::pair<int, int>> v;
    std::transform(retval.begin(), retval.end(), std::back_inserter(v), [last = 0](auto I) mutable { auto r = std::make_pair(last, I); last = I; return r; });

    return v;
}

Vocab<Frame> constructFrameVocabulary(const FileDatabase &database, unsigned int K, unsigned int speedinator)
{
    cv::Mat descriptors;

    for (auto video : database.listVideos())
    {
        auto v = database.loadVideo(video);
        auto &frames = v->frames();
        for (auto i = frames.begin(); i < frames.end(); i += speedinator)
            descriptors.push_back(i->descriptors);
    }
    cv::UMat copy;
    descriptors.copyTo(copy);

    return Vocab<Frame>(constructVocabulary(copy, K));
}

Vocab<SerializableScene> constructSceneVocabulary(const FileDatabase &database, unsigned int K, unsigned int speedinator)
{
    auto vocab = loadVocabulary<Vocab<Frame>>(database);
    if (!vocab)
    {
        throw std::runtime_error("trying to construct frame vocab but sift vocab is empty");
    }
    auto d = vocab->descriptors();

    cv::Mat descriptors;

    for (auto video : database.listVideos())
    {
        auto v = database.loadVideo(video);
        auto &frames = v->frames();
        for (auto i = frames.begin(); i < frames.end(); i += speedinator)
            descriptors.push_back(getFrameDescriptor(*i, d));
    }

    cv::UMat copy;
    descriptors.copyTo(copy);

    return Vocab<SerializableScene>(constructVocabulary(copy, K));
}

void overflow(std::vector<cv::Mat> &descriptor_levels, unsigned int K, unsigned int N, unsigned int level)
{
    // FIXME: better solution to not having enough points, maybe throw away?
    //while(descriptor_levels[level].rows < K){
    //    descriptor_levels[level].push_back();
    //}

    std::cout << "Enter overflow at level " << level << ", shape is: ";
    for (auto &a : descriptor_levels)
    {
        std::cout << a.rows << ">";
    }
    std::cout << std::endl;

    // throw away leftovers
    if (descriptor_levels[level].rows < K)
    {
        descriptor_levels[level] = cv::Mat();
        std::cout << "Early return" << std::endl;
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

    std::cout << "Exit overflow at level " << level << ", shape is: ";
    for (auto &a : descriptor_levels)
    {
        std::cout << a.rows << ">";
    }
    std::cout << std::endl;
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

    // collect features and reduce with kmeans as needed
    for (auto video : database.listVideos())
    {
        auto v = database.loadVideo(video);
        auto &frames = v->frames();
        for (auto i = frames.begin(); i < frames.end(); i += speedinator)
        {
            descriptor_levels[0].push_back(i->descriptors);
            // limit largest kmeans run
            if (descriptor_levels[0].rows >= N)
            {
                overflow(descriptor_levels, K, N, 0);
            }
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
    return Vocab<Frame>(descriptor_levels.back());
}

double boneheadedSimilarity(IVideo &v1, IVideo &v2, std::function<double(Frame, Frame)> comparator, SimilarityReporter reporter)
{
    auto frames1 = v1.frames();
    auto frames2 = v2.frames();

    double total = 0;

    auto len = std::min(frames1.size(), frames2.size());

    for (decltype(len) i = 0; i < len; i++)
    {
        auto t = comparator(frames1[i], frames2[i]);
        if (reporter)
            reporter(FrameSimilarityInfo{t, i, i,
                                         std::ref(v1), std::ref(v2)});

        total += (t != -1) ? t : 0;
    }

    return total / len;
}

std::optional<MatchInfo> findMatch(IVideo &target, FileDatabase &db)
{
    auto intcomp = [](auto f1, auto f2) { return cosineSimilarity(f1, f2) > 0.8 ? 3 : -3; };
    auto deref = [&target, &db](auto i) { return loadSceneDescriptor(i, target, db); };

    MatchInfo match{};
    std::vector<cv::Mat> targetScenes;
    boost::push_back(targetScenes, target.getScenes() | boost::adaptors::transformed(deref));

    for (auto v2 : db.listVideos())
    {
        if (v2 == target.name)
        {
            continue;
        }

        std::cout << "Calculating match for " << v2 << std::endl;
        std::vector<cv::Mat> knownScenes;
        auto v = db.loadVideo(v2);

        auto deref = [&v, &db](auto i) { return loadSceneDescriptor(i, *v, db); };
        boost::push_back(knownScenes, v->getScenes() | boost::adaptors::transformed(deref));

        auto &&alignments = calculateAlignment(knownScenes, targetScenes, intcomp, 3, 2);
        std::cout << targetScenes.size() << " " << knownScenes.size() << std::endl;
        if (alignments.size() > 0)
        {
            auto &a = alignments[0];
            std::cout << "Highest score: " << a.score << std::endl;
            if (a.score > match.matchConfidence)
            {
                match = MatchInfo{static_cast<double>(a.score), a.startKnown, a.endKnown, v2, alignments};
            }
        }
    }

    if (match.matchConfidence > 0.5)
    {
        return match;
    }

    return std::nullopt;
}
