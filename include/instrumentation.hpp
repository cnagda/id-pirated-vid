#ifndef INSTRUMENTATION_HPP
#define INSTRUMENTATION_HPP

// defines the interface for data flowing in and out of application
class InstrumentationHost {
public:
    // has side effects
    void exportData() const {}
}

class CSVExporter {
public:
    void exportInstrumentation(const InstrumentationHost& host) override;
}

#endif