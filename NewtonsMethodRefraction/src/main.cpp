#include "raylib.h"
#include "rlgl.h"
#include "raymath.h"
#include "external/glad.h"
#include "GPUTimer.h"
#include <iostream>

// When enabled, success rates for refractions or caustics will be shown onscreen. The impact on performance is very high, so this should be turned off unless you are interested in seeing the success rates.
// Regardless of whether this is enabled, the shaders will still tally success rates with atomic counters, and the success rate SSBO will be cleared each frame.
// For maximum performance, those parts should be removed manually.
#define READ_SUCCESS_STATS 0

// --- Constants ---
constexpr int shadowMapResolution = 2048;
constexpr int causticsResolution = 200;

// --- Function Declarations ---
void GenerateShadowmap(unsigned int& shadowFBO, unsigned int& shadowDepthTex, unsigned int& shadowNormalTex);
void UnloadShadowmap(unsigned int& shadowFBO, unsigned int& shadowDepthTex, unsigned int& shadowNormalTex);
Texture2D TextureFromId(unsigned int id, int w, int h);
void DrawScene(const Model& pool);
void SetupCamera(Camera& camera, int preset);
void SetupWaterMaterial(Model& plane, Texture2D colorTex, Texture2D normalTex, Texture2D depthTex, Texture waterNormals1, Texture waterNormals2, TextureCubemap skybox, int screenWidth, int screenHeight, Shader& waterShader);
void SetupPoolMaterial(Model& pool, Texture poolTexture, Texture shadowDepthTex, Texture causticsTexture, Shader& opaqueShader);
void SetupCausticsMaterial(Model& causticsPlane, Texture shadowDepthTex, Texture shadowNormalTex, Texture waterNormals1, Texture waterNormals2, Shader& causticsShader);

// --- Camera Presets ---
Vector3 cameraPositions[] = {
    Vector3{ 2.88537574f, 16.0f, 12.5140343f },
    Vector3{ 13.4655485f, 16.0f, -7.26100874f }
};
Vector3 cameraTargets[] = {
    Vector3{ 2.10622692f, -0.0122699738f, 12.4981375f },
    Vector3{ 3.47687912f, 7.09196472f, 1.56359339f }
};

// --- Main Entry Point ---
int main()
{
    // --- Window and Camera Initialization ---
    const int screenWidth = 2000;
    const int screenHeight = 1600;
    InitWindow(screenWidth, screenHeight, "Water Refraction Test Scene");

    Camera camera = { 0 };
    SetupCamera(camera, 0); // Use first preset

    // --- Create Main Scene FBO ---
    unsigned int fbo = 0, colorTex = 0, normalTex = 0, depthTex = 0;
    {
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);

        // Color buffer (RGBA8)
        glGenTextures(1, &colorTex);
        glBindTexture(GL_TEXTURE_2D, colorTex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, screenWidth, screenHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorTex, 0);

        // Normal buffer (RGB32F)
        glGenTextures(1, &normalTex);
        glBindTexture(GL_TEXTURE_2D, normalTex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, screenWidth, screenHeight, 0, GL_RGB, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, normalTex, 0);

        // Depth buffer (Depth24)
        glGenTextures(1, &depthTex);
        glBindTexture(GL_TEXTURE_2D, depthTex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, screenWidth, screenHeight, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        SetTextureWrap(TextureFromId(depthTex, screenWidth, screenHeight), TEXTURE_WRAP_CLAMP);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthTex, 0);

        // Set draw buffers
        GLenum drawBuffers[2] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
        glDrawBuffers(2, drawBuffers);

        // Check FBO completeness
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            TraceLog(LOG_ERROR, "FBO is not complete!");
        }
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    // --- Shadowmap FBO (for sun shadows) ---
    unsigned int shadowFbo = 0, shadowDepthTex = 0, shadowNormalTex = 0;
    GenerateShadowmap(shadowFbo, shadowDepthTex, shadowNormalTex);

    // --- Load Models, Textures, and Shaders ---
    Model pool = LoadModel("resources/models/pool.glb");
    Shader opaqueShader = LoadShader("resources/shaders/opaque.vert", "resources/shaders/opaque.frag");
    Texture poolTexture = LoadTexture("resources/models/checkerboard_pattern.png");

    int camPosLoc = GetShaderLocation(opaqueShader, "camPos");
    int shadowVPLoc = GetShaderLocation(opaqueShader, "shadowVP");

    BoundingBox poolBounds = GetModelBoundingBox(pool);
    Model plane = LoadModelFromMesh(GenMeshPlane(poolBounds.max.x - poolBounds.min.x, poolBounds.max.z - poolBounds.min.z, 1, 1));

    Texture waterNormals1 = LoadTexture("resources/models/waterNormals1.png");
    Texture waterNormals2 = LoadTexture("resources/models/waterNormals2.png");
    SetTextureFilter(waterNormals1, TEXTURE_FILTER_BILINEAR);
    SetTextureFilter(waterNormals2, TEXTURE_FILTER_BILINEAR);
    SetTextureWrap(waterNormals1, TEXTURE_WRAP_MIRROR_REPEAT);
    SetTextureWrap(waterNormals2, TEXTURE_WRAP_MIRROR_REPEAT);

    Image cubemapImage = LoadImage("resources/models/StandardCubeMap.png");
    TextureCubemap skybox = LoadTextureCubemap(cubemapImage, CUBEMAP_LAYOUT_CROSS_FOUR_BY_THREE);
    UnloadImage(cubemapImage);

    Mesh cube = GenMeshCube(1.0f, 1.0f, 1.0f);
    Model skyboxModel = LoadModelFromMesh(cube);
    Shader skyboxShader = LoadShader("resources/shaders/skybox.vert", "resources/shaders/skybox.frag");
    skyboxModel.materials[0].shader = skyboxShader;
    int envMap = MATERIAL_MAP_CUBEMAP;
    SetShaderValue(skyboxShader, GetShaderLocation(skyboxShader, "environmentMap"), &envMap, SHADER_UNIFORM_INT);
    bool doGamma = false, vFlipped = false;
    SetShaderValue(skyboxShader, GetShaderLocation(skyboxShader, "doGamma"), &doGamma, SHADER_UNIFORM_INT);
    SetShaderValue(skyboxShader, GetShaderLocation(skyboxShader, "vFlipped"), &vFlipped, SHADER_UNIFORM_INT);
    skyboxModel.materials[0].maps[MATERIAL_MAP_CUBEMAP].texture = skybox;
    skyboxShader.locs[SHADER_LOC_MAP_CUBEMAP] = GetShaderLocation(skyboxShader, "environmentMap");

    Shader waterShader = LoadShader("resources/shaders/opaque.vert", "resources/shaders/water.frag");
    waterShader.locs[SHADER_LOC_MAP_ALBEDO] = GetShaderLocation(waterShader, "sceneColors");
    waterShader.locs[SHADER_LOC_MAP_NORMAL] = GetShaderLocation(waterShader, "texNormal");
    waterShader.locs[SHADER_LOC_MAP_EMISSION] = GetShaderLocation(waterShader, "texNormal2");
    waterShader.locs[SHADER_LOC_MAP_SPECULAR] = GetShaderLocation(waterShader, "sceneNormals");
    waterShader.locs[SHADER_LOC_MAP_HEIGHT] = GetShaderLocation(waterShader, "sceneDepth");
    waterShader.locs[SHADER_LOC_MAP_CUBEMAP] = GetShaderLocation(waterShader, "skybox");

    int waterShaderCamPosLoc = GetShaderLocation(waterShader, "camPos");
    int invViewLoc = GetShaderLocation(waterShader, "invView");
    int invProjectionLoc = GetShaderLocation(waterShader, "invProjection");
    int viewProjectionLoc = GetShaderLocation(waterShader, "viewProjection");
    int timeLoc = GetShaderLocation(waterShader, "time");

    // --- Sun Direction and Shadow Camera ---
    Vector3 sunDirection = Vector3Normalize(Vector3{ -0.5f, -1.0f, 0.5f });
    Camera shadowCamera = { 0 };
    shadowCamera.position = Vector3Scale(Vector3{ -sunDirection.x, -sunDirection.y, -sunDirection.z }, 100);
    shadowCamera.target = Vector3{ 0.0f, 0.0f, 0.0f };
    shadowCamera.up = Vector3{ 0.0f, 0.0f, 1.0f };
    shadowCamera.fovy = 70.0f;
    shadowCamera.projection = CAMERA_ORTHOGRAPHIC;

    Shader shadowShader = LoadShader("resources/shaders/opaque.vert", "resources/shaders/shadow.frag");
    opaqueShader.locs[SHADER_LOC_MAP_HEIGHT] = GetShaderLocation(opaqueShader, "shadowMap");

    // --- Caustics FBO and Plane ---
    Model causticsPlane = LoadModelFromMesh(GenMeshPlane(poolBounds.max.x - poolBounds.min.x, poolBounds.max.z - poolBounds.min.z, causticsResolution, causticsResolution));
    unsigned int causticsFBO = 0, causticsTexture = 0, causticsDepthRBO = 0;
    {
        glGenFramebuffers(1, &causticsFBO);
        glBindFramebuffer(GL_FRAMEBUFFER, causticsFBO);

        glGenTextures(1, &causticsTexture);
        glBindTexture(GL_TEXTURE_2D, causticsTexture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, shadowMapResolution, shadowMapResolution, 0, GL_RED, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, causticsTexture, 0);

        glGenRenderbuffers(1, &causticsDepthRBO);
        glBindRenderbuffer(GL_RENDERBUFFER, causticsDepthRBO);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, shadowMapResolution, shadowMapResolution);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, causticsDepthRBO);

        GLenum causticsDrawBuffers[1] = { GL_COLOR_ATTACHMENT0 };
        glDrawBuffers(1, causticsDrawBuffers);

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            TraceLog(LOG_ERROR, "FBO is not complete!");
        }
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    Shader causticsShader = LoadShader("resources/shaders/caustics.vert", "resources/shaders/caustics.frag");
    causticsShader.locs[SHADER_LOC_MAP_ALBEDO] = GetShaderLocation(causticsShader, "depthBuffer");
    causticsShader.locs[SHADER_LOC_MAP_OCCLUSION] = GetShaderLocation(causticsShader, "normalBuffer");
    causticsShader.locs[SHADER_LOC_MAP_NORMAL] = GetShaderLocation(causticsShader, "texNormal");
    causticsShader.locs[SHADER_LOC_MAP_HEIGHT] = GetShaderLocation(causticsShader, "texNormal2");
    int causticsInvProjectionLoc = GetShaderLocation(causticsShader, "invProjection");
    int causticsMatVPLoc = GetShaderLocation(causticsShader, "matVP");
    int causticsInvViewLoc = GetShaderLocation(causticsShader, "invView");
    int causticsTimeLoc = GetShaderLocation(causticsShader, "time");
    int causticsSunPosLoc = GetShaderLocation(causticsShader, "sunPos");

    // --- Material Setup ---
    SetupWaterMaterial(plane, TextureFromId(colorTex, screenWidth, screenHeight), TextureFromId(normalTex, screenWidth, screenHeight), TextureFromId(depthTex, screenWidth, screenHeight), waterNormals1, waterNormals2, skybox, screenWidth, screenHeight, waterShader);
    SetupPoolMaterial(pool, poolTexture, TextureFromId(shadowDepthTex, shadowMapResolution, shadowMapResolution), TextureFromId(causticsTexture, shadowMapResolution, shadowMapResolution), opaqueShader);
    SetupCausticsMaterial(causticsPlane, TextureFromId(shadowDepthTex, shadowMapResolution, shadowMapResolution), TextureFromId(shadowNormalTex, shadowMapResolution, shadowMapResolution), waterNormals1, waterNormals2, causticsShader);
    opaqueShader.locs[SHADER_LOC_MAP_EMISSION] = GetShaderLocation(opaqueShader, "causticsMap");

    // --- SSBO for Pixel Stats ---
    GLuint ssbo;
    glGenBuffers(1, &ssbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(GLuint) * 2, nullptr, GL_DYNAMIC_COPY);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo);

    SetTargetFPS(60);
    DisableCursor();

    GPUTimer gpuTimer;
    bool trackingTime = false;
    double accumulatedTime = 0.0;
    int accumulatedFrames = 0;

    // --- Main Render Loop ---
    while (!WindowShouldClose())
    {
        // --- Camera Preset Switching ---
        if (IsKeyPressed(KEY_ONE)) SetupCamera(camera, 0);
        else if (IsKeyPressed(KEY_TWO)) SetupCamera(camera, 1);

        // The first time the T key is pressed, reset the timer and start tracking time
        // The second time the T key is pressed, stop tracking time and report the average frame time
        if (IsKeyPressed(KEY_T))
        {
            if (!trackingTime)
            {
                accumulatedTime = 0.0;
                accumulatedFrames = 0;
                trackingTime = true;
            }
            else
            {
                trackingTime = false;
                if (accumulatedFrames > 0) std::cout << "Average frame time: " << (accumulatedTime / accumulatedFrames) << " ms over " << accumulatedFrames << " frames." << std::endl;
            }
        }

        // --- Update Shader Uniforms ---
        SetShaderValue(opaqueShader, camPosLoc, &camera.position, SHADER_UNIFORM_VEC3);
        SetShaderValue(waterShader, waterShaderCamPosLoc, &camera.position, SHADER_UNIFORM_VEC3);
        float time = GetTime();
        SetShaderValue(waterShader, timeLoc, &time, SHADER_UNIFORM_FLOAT);
        SetShaderValue(causticsShader, causticsSunPosLoc, &shadowCamera.position, SHADER_UNIFORM_VEC3);

        // --- Reset SSBO Counters ---
#if READ_SUCCESS_STATS
        GLuint zeroes[2] = { 0, 0 };
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(zeroes), zeroes);
#endif // READ_SUCCESS_STATS

        // --- Render Shadowmap ---
        glBindFramebuffer(GL_FRAMEBUFFER, shadowFbo);
        glViewport(0, 0, shadowMapResolution, shadowMapResolution);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        BeginMode3D(shadowCamera);
        Matrix shadowView = rlGetMatrixModelview();
        Matrix shadowProjection = rlGetMatrixProjection();
        Matrix shadowVP = MatrixMultiply(shadowView, shadowProjection);
        SetShaderValueMatrix(opaqueShader, shadowVPLoc, shadowVP);
        for (int i = 0; i < pool.materialCount; i++) pool.materials[i].shader = shadowShader;
        DrawScene(pool);
        EndMode3D();
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        // --- Render Caustics ---
        Matrix shadowInvProjection = MatrixInvert(shadowProjection);
        Matrix shadowInvView = MatrixInvert(shadowView);
        SetShaderValueMatrix(causticsShader, causticsInvProjectionLoc, shadowInvProjection);
        SetShaderValueMatrix(causticsShader, causticsMatVPLoc, shadowVP);
        SetShaderValueMatrix(causticsShader, causticsInvViewLoc, shadowInvView);
        SetShaderValue(causticsShader, causticsTimeLoc, &time, SHADER_UNIFORM_FLOAT);
        glBindFramebuffer(GL_FRAMEBUFFER, causticsFBO);
        glViewport(0, 0, shadowMapResolution, shadowMapResolution);
        glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        BeginMode3D(shadowCamera);
        DrawModel(causticsPlane, Vector3{ 0.0f, 0.0f, 0.0f }, 1.0f, WHITE);
        EndMode3D();
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        // --- Render Opaque Scene to FBO ---
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glViewport(0, 0, screenWidth, screenHeight);
        glClearColor(0.1f, 0.1f, 0.1f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        BeginMode3D(camera);
        Matrix view = rlGetMatrixModelview();
        Matrix projection = rlGetMatrixProjection();
        Matrix invView = MatrixInvert(view);
        Matrix invProjection = MatrixInvert(projection);
        Matrix viewProjection = MatrixMultiply(view, projection);
        SetShaderValueMatrix(waterShader, invViewLoc, invView);
        SetShaderValueMatrix(waterShader, invProjectionLoc, invProjection);
        SetShaderValueMatrix(waterShader, viewProjectionLoc, viewProjection);
        for (int i = 0; i < pool.materialCount; i++) pool.materials[i].shader = opaqueShader;
        DrawScene(pool);
        EndMode3D();
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        // --- Final Screen Rendering ---
        BeginDrawing();
        ClearBackground(RAYWHITE);
        BeginMode3D(camera);
        rlDisableBackfaceCulling();
        rlDisableDepthMask();
        DrawModel(skyboxModel, Vector3{ 0, 0, 0 }, 1.0f, WHITE);
        rlEnableBackfaceCulling();
        rlEnableDepthMask();
        EndMode3D();

        // Blit FBO depth to main buffer
        glBlitNamedFramebuffer(fbo, 0, 0, 0, screenWidth, screenHeight, 0, 0, screenWidth, screenHeight, GL_DEPTH_BUFFER_BIT, GL_NEAREST);

        // Show color buffer
        DrawTexturePro(TextureFromId(colorTex, screenWidth, screenHeight), Rectangle{ 0.0f, 0.0f, screenWidth, -screenHeight }, Rectangle{ 0.0f, 0.0f, screenWidth, screenHeight }, Vector2{ 0.0f, 0.0f }, 0.0f, WHITE);

        if (trackingTime)
        {
            gpuTimer.Begin();
        }
        BeginMode3D(camera);
        DrawModel(plane, Vector3{ 0.0f, 0.0f, 0.0f }, 1.0f, WHITE);
        EndMode3D();
        if (trackingTime)
        {
            GLint64 elapsedTime = gpuTimer.End(); // Nanoseconds
            accumulatedTime += (double)elapsedTime / 1000000.0;
            accumulatedFrames++;
        }

        // Show pixel stats from SSBO
#if(READ_SUCCESS_STATS)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
        GLuint* data = (GLuint*)glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLuint) * 2, GL_MAP_READ_BIT);
        if (data)
        {
            float percentSuccess = ((float)data[0] / (data[0] + data[1])) * 100.0f;
            DrawText(TextFormat("Success: %u, Failure: %u (%.2f%%)", data[0], data[1], percentSuccess), 10, 10, 50, RED);
            glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
        }
#endif

        // Screenshot
        if (IsKeyPressed(KEY_K))
        {
            TakeScreenshot("screenshot.png");
            TraceLog(LOG_INFO, "Screenshot saved as screenshot.png");
        }
        EndDrawing();
    }

    // --- Cleanup ---
    glDeleteTextures(1, &colorTex);
    glDeleteTextures(1, &normalTex);
    glDeleteTextures(1, &depthTex);
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &causticsTexture);
    glDeleteFramebuffers(1, &causticsFBO);
    glDeleteRenderbuffers(1, &causticsDepthRBO);
    UnloadShader(causticsShader);
    UnloadModel(causticsPlane);
    UnloadModel(pool);
    UnloadShader(opaqueShader);
    UnloadTexture(poolTexture);
    UnloadModel(plane);
    UnloadModel(skyboxModel);
    UnloadShader(skyboxShader);
    UnloadTexture(skybox);
    UnloadTexture(waterNormals1);
    UnloadTexture(waterNormals2);
    UnloadShader(waterShader);
    UnloadShadowmap(shadowFbo, shadowDepthTex, shadowNormalTex);
    UnloadShader(shadowShader);
    CloseWindow();
    return 0;
}

// --- Function Definitions ---

// Setup camera from preset
void SetupCamera(Camera& camera, int preset)
{
    camera.position = cameraPositions[preset];
    camera.target = cameraTargets[preset];
    camera.up = Vector3{ 0.0f, 1.0f, 0.0f };
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;
}

// Setup water material for the water plane
void SetupWaterMaterial(Model& plane, Texture2D colorTex, Texture2D normalTex, Texture2D depthTex, Texture waterNormals1, Texture waterNormals2, TextureCubemap skybox, int screenWidth, int screenHeight, Shader& waterShader)
{
    for (int i = 0; i < plane.materialCount; i++) {
        plane.materials[i].shader = waterShader;
        plane.materials[i].maps[MATERIAL_MAP_DIFFUSE].texture = colorTex;
        plane.materials[i].maps[MATERIAL_MAP_NORMAL].texture = waterNormals1;
        plane.materials[i].maps[MATERIAL_MAP_EMISSION].texture = waterNormals2;
        plane.materials[i].maps[MATERIAL_MAP_SPECULAR].texture = normalTex;
        plane.materials[i].maps[MATERIAL_MAP_HEIGHT].texture = depthTex;
        plane.materials[i].maps[MATERIAL_MAP_CUBEMAP].texture = skybox;
    }
}

// Setup pool material for the pool model
void SetupPoolMaterial(Model& pool, Texture poolTexture, Texture shadowDepthTex, Texture causticsTexture, Shader& opaqueShader)
{
    for (int i = 0; i < pool.materialCount; i++)
    {
        pool.materials[i].shader = opaqueShader;
        pool.materials[i].maps[MATERIAL_MAP_DIFFUSE].texture = poolTexture;
        pool.materials[i].maps[MATERIAL_MAP_HEIGHT].texture = shadowDepthTex;
        pool.materials[i].maps[MATERIAL_MAP_EMISSION].texture = causticsTexture;
    }
}

// Setup caustics material for the caustics plane
void SetupCausticsMaterial(Model& causticsPlane, Texture shadowDepthTex, Texture shadowNormalTex, Texture waterNormals1, Texture waterNormals2, Shader& causticsShader)
{
    causticsPlane.materials[0].shader = causticsShader;
    causticsPlane.materials[0].maps[MATERIAL_MAP_ALBEDO].texture = shadowDepthTex;
    causticsPlane.materials[0].maps[MATERIAL_MAP_OCCLUSION].texture = shadowNormalTex;
    causticsPlane.materials[0].maps[MATERIAL_MAP_NORMAL].texture = waterNormals1;
    causticsPlane.materials[0].maps[MATERIAL_MAP_HEIGHT].texture = waterNormals2;
}

// Generate a shadowmap framebuffer with depth and normal textures
void GenerateShadowmap(unsigned int& shadowFBO, unsigned int& shadowDepthTex, unsigned int& shadowNormalTex)
{
    glGenFramebuffers(1, &shadowFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, shadowFBO);
    // Depth texture
    glGenTextures(1, &shadowDepthTex);
    glBindTexture(GL_TEXTURE_2D, shadowDepthTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, shadowMapResolution, shadowMapResolution, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, shadowDepthTex, 0);
    // Normal texture (RGB32F)
    glGenTextures(1, &shadowNormalTex);
    glBindTexture(GL_TEXTURE_2D, shadowNormalTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, shadowMapResolution, shadowMapResolution, 0, GL_RGB, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, shadowNormalTex, 0);
    GLenum drawBuffers[1] = { GL_COLOR_ATTACHMENT1 };
    glDrawBuffers(1, drawBuffers);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        TraceLog(LOG_ERROR, "Shadow FBO is not complete!");
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

// Unload shadowmap resources
void UnloadShadowmap(unsigned int& shadowFBO, unsigned int& shadowDepthTex, unsigned int& shadowNormalTex)
{
    glDeleteFramebuffers(1, &shadowFBO);
    glDeleteTextures(1, &shadowDepthTex);
    glDeleteTextures(1, &shadowNormalTex);
    shadowFBO = 0;
    shadowDepthTex = 0;
    shadowNormalTex = 0;
}

// Draw the pool model at the origin
void DrawScene(const Model& pool)
{
    DrawModel(pool, Vector3{ 0.0f, 0.0f, 0.0f }, 1.0f, WHITE);
}

// Wrap OpenGL texture as raylib Texture2D for DrawTexture
Texture2D TextureFromId(unsigned int id, int w, int h) {
    Texture2D tex = { 0 };
    tex.id = id;
    tex.width = w;
    tex.height = h;
    tex.mipmaps = 1;
    tex.format = PIXELFORMAT_UNCOMPRESSED_R8G8B8; // For colorTex
    return tex;
}
