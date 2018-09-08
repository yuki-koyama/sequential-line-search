#ifndef DRAWUTILITY_H
#define DRAWUTILITY_H

#include <string>
#include <vector>
#ifdef __APPLE__
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif

namespace DrawUtility
{
    int  loadShader(std::string vtxShdName, std::string frgShdName, GLuint *lpProg);
    void printShaderLog(GLuint shader);
    void printProgramInfoLog(GLuint program);
}

#endif // DRAWUTILITY_H
