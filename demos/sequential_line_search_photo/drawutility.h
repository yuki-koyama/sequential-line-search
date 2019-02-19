#ifndef DRAWUTILITY_H
#define DRAWUTILITY_H

#include <string>
#include <vector>
#ifdef __APPLE__
#include <OpenGL/gl.h>
#else
#ifdef WIN32
#include <GL/glew.h> // to use shader. Windows only support GL ~1.1
#endif
#include <GL/gl.h>
#endif

namespace DrawUtility
{
    int  loadShader(std::string vtxShdName, std::string frgShdName, GLuint *lpProg);
    void printShaderLog(GLuint shader);
    void printProgramInfoLog(GLuint program);
}

#endif // DRAWUTILITY_H
