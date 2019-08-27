#ifndef directoryutility_h
#define directoryutility_h

#include <QApplication>
#include <QDir>
#include <ctime>
#include <iomanip>
#include <string>

namespace DirectoryUtility
{
    inline std::string getCurrentTimeInString()
    {
        const std::time_t t = std::time(nullptr);
        std::stringstream s;
        s << std::put_time(std::localtime(&t), "%Y%m%d%H%M%S");
        return s.str();
    }

    inline std::string getResourceDirectory()
    {
        return QCoreApplication::applicationDirPath().toStdString() + "/../Resources";
    }

    inline void createDirectoryIfNotExist(const std::string& dirPath)
    {
        QDir dir(QString::fromStdString(dirPath));
        if (!dir.exists())
        {
            dir.mkpath(".");
        }
    }

    inline std::string getTemporaryDirectory()
    {
        const std::string baseDirPath = QCoreApplication::applicationDirPath().toStdString();
        const std::string mainDirPath = baseDirPath + "/" + getCurrentTimeInString();
        createDirectoryIfNotExist(mainDirPath);
        return mainDirPath;
    }
} // namespace DirectoryUtility

#endif /* directoryutility_h */
