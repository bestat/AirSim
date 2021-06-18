// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef air_VehicleSimApiBase_hpp
#define air_VehicleSimApiBase_hpp

#include "common/CommonStructs.hpp"
#include "common/UpdatableObject.hpp"
#include "common/ImageCaptureBase.hpp"
#include "physics/Kinematics.hpp"
#include "physics/Environment.hpp"
#include "common/AirSimSettings.hpp"

namespace msr
{
namespace airlib
{

    class VehicleSimApiBase : public msr::airlib::UpdatableObject
    {
    public:
        virtual ~VehicleSimApiBase() = default;

        virtual void update() override
        {
            UpdatableObject::update();
        }

        //this method is called at every render tick when we want to transfer state from
        //physics engine to render engine. As physics engine is halted while
        //this happens, this method should do minimal processing
        virtual void updateRenderedState(float dt)
        {
            unused(dt);
            //derived class should override if needed
        }
        //called when render changes are required at every render tick
        virtual void updateRendering(float dt)
        {
            unused(dt);
            //derived class should override if needed
        }

        virtual const ImageCaptureBase* getImageCapture() const = 0;
        virtual ImageCaptureBase* getImageCapture()
        {
            return const_cast<ImageCaptureBase*>(static_cast<const VehicleSimApiBase*>(this)->getImageCapture());
        }

        virtual void initialize() = 0;

        virtual std::vector<ImageCaptureBase::ImageResponse> getImages(const std::vector<ImageCaptureBase::ImageRequest>& request) const = 0;
        virtual std::vector<uint8_t> getImage(const std::string& camera_name, ImageCaptureBase::ImageType image_type) const = 0;

        virtual Pose getPose() const = 0;
        virtual void setPose(const Pose& pose, bool ignore_collision) = 0;
        virtual const Kinematics::State* getGroundTruthKinematics() const = 0;
        virtual const msr::airlib::Environment* getGroundTruthEnvironment() const = 0;

        virtual CameraInfo getCameraInfo(const std::string& camera_name) const = 0;
        virtual void setCameraPose(const std::string& camera_name, const Pose& pose) = 0;
        virtual void setCameraFoV(const std::string& camera_name, float fov_degrees) = 0;
		virtual void setCameraImageSize(const std::string& camera_name, const float width, const float height) = 0;
		virtual void setCameraPostProcess(
			const std::string& camera_name,
			const float blend_weight,
			const float bloom_intensity,
			const int auto_exposure_method, const float auto_exposure_bias,
			const float auto_exposure_speed_up, const float auto_exposure_speed_down,
			const float auto_exposure_max_brightness, const float auto_exposure_min_brightness,
			const float lens_flare_intensity,
			const float ambient_occlusion_intensity, const float ambient_occlusion_radius,
			const std::vector<float>& indirect_lighting_color_rgb, const float indirect_lighting_intensity,
			const float motion_blur_amount
		) = 0;
        virtual void setDistortionParam(const std::string& camera_name, const std::string& param_name, float value) = 0;
        virtual std::vector<float> getDistortionParams(const std::string& camera_name) = 0;

        virtual CollisionInfo getCollisionInfo() const = 0;
        virtual int getRemoteControlID() const = 0; //which RC to use, 0 is first one, -1 means disable RC (use keyborad)
        virtual RCData getRCData() const = 0; //get reading from RC from simulator's host OS
        virtual std::string getVehicleName() const = 0;
        virtual std::string getRecordFileLine(bool is_header_line) const = 0;
        virtual void toggleTrace() = 0;
        virtual void setTraceLine(const std::vector<float>& color_rgba, float thickness) = 0;

        //use pointer here because of derived classes for VehicleSetting
        const AirSimSettings::VehicleSetting* getVehicleSetting() const
        {
            return AirSimSettings::singleton().getVehicleSetting(getVehicleName());
        }
    };
}
} //namespace
#endif
