#ifndef DRAWUTILITY_H
#define DRAWUTILITY_H

#include <string>
#include <vector>
#include <OpenGL/gl.h>

namespace DrawUtility
{
    int  loadShader(std::string vtxShdName, std::string frgShdName, GLuint *lpProg);
    void printShaderLog(GLuint shader);
    void printProgramInfoLog(GLuint program);
}

#endif // DRAWUTILITY_H
