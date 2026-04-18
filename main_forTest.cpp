#include "Problem/ApplicationForTest.h"

// ── 문제 파일 include ─────────────────────────────────────────────────────────
// 풀고 싶은 문제를 include하고 아래 addFeature()에 등록하세요.
// 정답을 확인하려면 1_MakeTriangle_Solv.h 를 함께 include하세요.

#include "Problem/1_MakeTriangle.h"
#include "Problem/1_MakeTriangle_Solv.h"

#include <iostream>

int main()
{
    try
    {
        Application app(1280, 720, "Vision3D - Problem Runner");

        // ── 문제 / 정답 Feature 등록 ──────────────────────────────────────────
        app.addFeature(std::make_unique<MakeTriangle>());
        app.addFeature(std::make_unique<MakeTriangleSolv>());

        app.run();
    }
    catch (const std::exception &e)
    {
        std::cerr << "[ERROR] " << e.what() << '\n';
        return 1;
    }
    return 0;
}
