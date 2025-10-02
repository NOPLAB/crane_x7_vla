FROM ros:humble-ros-base-jammy AS base

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y --no-install-recommends \
    xserver-xorg

RUN apt update && apt install -y --no-install-recommends \
    adwaita-icon-theme alsa-topology-conf alsa-ucm-conf at-spi2-core autoconf automake \
    autotools-dev avahi-daemon bind9-host bind9-libs blt comerr-dev dbus-user-session \
    dconf-gsettings-backend dconf-service dmsetup file fonts-lato fonts-lyx gdal-data geoclue-2.0 \
    gfortran gfortran-11 gir1.2-glib-2.0 glib-networking glib-networking-common \
    glib-networking-services gsettings-desktop-schemas gstreamer1.0-plugins-base \
    gtk-update-icon-cache hicolor-icon-theme humanity-icon-theme i965-va-driver ibverbs-providers \
    icu-devtools ignition-tools ignition-transport11-cli iio-sensor-proxy intel-media-va-driver \
    iso-codes javascript-common krb5-multidev libaacs0 libaec0 libaom3 libargon2-1 libarmadillo10 \
    libarpack2 libasound2 libasound2-data libass9 libassimp-dev libassimp5 libasyncns0 \
    libatk-bridge2.0-0 libatk1.0-0 libatk1.0-data libatspi2.0-0 libavahi-client3 \
    libavahi-common-data libavahi-common3 libavahi-core7 libavahi-glib1 libavc1394-0 \
    libavcodec-dev libavcodec58 libavdevice-dev libavdevice58 libavfilter-dev libavfilter7 \
    libavformat-dev libavformat58 libavutil-dev libavutil56 libbdplus0 libblkid-dev libblosc1 \
    libbluray2 libboost-all-dev libboost-atomic-dev libboost-atomic1.74-dev libboost-atomic1.74.0 \
    libboost-chrono-dev libboost-chrono1.74-dev libboost-chrono1.74.0 libboost-container-dev \
    libboost-container1.74-dev libboost-container1.74.0 libboost-context-dev \
    libboost-context1.74-dev libboost-context1.74.0 libboost-coroutine-dev \
    libboost-coroutine1.74-dev libboost-coroutine1.74.0 libboost-date-time-dev \
    libboost-date-time1.74-dev libboost-date-time1.74.0 libboost-exception-dev \
    libboost-exception1.74-dev libboost-fiber-dev libboost-fiber1.74-dev libboost-fiber1.74.0 \
    libboost-filesystem-dev libboost-filesystem1.74-dev libboost-filesystem1.74.0 \
    libboost-graph-dev libboost-graph-parallel-dev libboost-graph-parallel1.74-dev \
    libboost-graph-parallel1.74.0 libboost-graph1.74-dev libboost-graph1.74.0 \
    libboost-iostreams-dev libboost-iostreams1.74-dev libboost-iostreams1.74.0 libboost-locale-dev \
    libboost-locale1.74-dev libboost-locale1.74.0 libboost-log-dev libboost-log1.74-dev \
    libboost-log1.74.0 libboost-math-dev libboost-math1.74-dev libboost-math1.74.0 \
    libboost-mpi-dev libboost-mpi-python-dev libboost-mpi-python1.74-dev libboost-mpi-python1.74.0 \
    libboost-mpi1.74-dev libboost-mpi1.74.0 libboost-nowide-dev libboost-nowide1.74-dev \
    libboost-nowide1.74.0 libboost-numpy-dev libboost-numpy1.74-dev libboost-numpy1.74.0 \
    libboost-program-options-dev libboost-program-options1.74-dev libboost-program-options1.74.0 \
    libboost-python-dev libboost-python1.74-dev libboost-python1.74.0 libboost-random-dev \
    libboost-random1.74-dev libboost-random1.74.0 libboost-regex-dev libboost-regex1.74-dev \
    libboost-regex1.74.0 libboost-serialization-dev libboost-serialization1.74-dev \
    libboost-serialization1.74.0 libboost-stacktrace-dev libboost-stacktrace1.74-dev \
    libboost-stacktrace1.74.0 libboost-system-dev libboost-system1.74-dev libboost-system1.74.0 \
    libboost-test-dev libboost-test1.74-dev libboost-test1.74.0 libboost-thread-dev \
    libboost-thread1.74-dev libboost-thread1.74.0 libboost-timer-dev libboost-timer1.74-dev \
    libboost-timer1.74.0 libboost-tools-dev libboost-type-erasure-dev \
    libboost-type-erasure1.74-dev libboost-type-erasure1.74.0 libboost-wave-dev \
    libboost-wave1.74-dev libboost-wave1.74.0 libboost1.74-tools-dev libbrotli-dev libbs2b0 \
    libbsd-dev libcaca0 libcaf-openmpi-3 libcairo-gobject2 libcap2-bin libcbor0.8 libccd-dev \
    libccd2 libcdio-cdda2 libcdio-paranoia2 libcdio19 libcdparanoia0 libcfitsio9 libcharls2 \
    libchromaprint1 libclang1-14 libcoarrays-dev libcoarrays-openmpi-dev libcodec2-1.0 libcolord2 \
    libcryptsetup12 libcups2 libcurl4-openssl-dev libdaemon0 libdart-collision-bullet-dev \
    libdart-collision-bullet6.12 libdart-collision-ode-dev libdart-collision-ode6.12 libdart-dev \
    libdart-external-convhull-3d-dev libdart-external-ikfast-dev libdart-external-odelcpsolver-dev \
    libdart-external-odelcpsolver6.12 libdart-utils-dev libdart-utils-urdf-dev \
    libdart-utils-urdf6.12 libdart-utils6.12 libdart6.12 libdav1d5 libdc1394-25 libdc1394-dev \
    libdconf1 libde265-0 libdecor-0-0 libdecor-0-plugin-1-cairo libdeflate-dev libdevmapper1.02.1 \
    libdouble-conversion3 libdraco-dev libdraco4 libegl-dev libevent-2.1-7 libevent-dev \
    libevent-extra-2.1-7 libevent-openssl-2.1-7 libevent-pthreads-2.1-7 libexif-dev libexif-doc \
    libexif12 libfabric1 libfcl-dev libfcl0.7 libffi-dev libfido2-1 libflac8 libflite1 \
    libfreeimage3 libfreetype-dev libfreetype6-dev libfreexl1 libfyba0 libgdal30 libgdcm-dev \
    libgdcm3.0 libgdk-pixbuf-2.0-0 libgdk-pixbuf2.0-bin libgdk-pixbuf2.0-common libgeos-c1v5 \
    libgeos3.10.2 libgeotiff5 libgflags-dev libgflags2.2 libgfortran-11-dev libgif7 \
    libgirepository-1.0-1 libgl-dev libgl1-mesa-dev libgl2ps1.4 libgles-dev libgles1 libgles2 \
    libglew-dev libglew2.2 libglib2.0-bin libglib2.0-data libglib2.0-dev libglib2.0-dev-bin \
    libglu1-mesa libglu1-mesa-dev libglvnd-core-dev libglvnd-dev libglx-dev libgme0 libgphoto2-6 \
    libgphoto2-dev libgphoto2-l10n libgphoto2-port12 libgsm1 libgssrpc4 \
    libgstreamer-plugins-base1.0-0 libgstreamer1.0-0 libgtk-3-0 libgtk-3-bin libgtk-3-common \
    libgts-dev libhdf4-0-alt libhdf5-103-1 libhdf5-hl-100 libheif1 libhwloc-dev libhwloc-plugins \
    libhwloc15 libibverbs-dev libibverbs1 libice-dev libicu-dev libiec61883-0 libigdgmm12 \
    libignition-cmake2-dev libignition-common4 libignition-common4-av libignition-common4-av-dev \
    libignition-common4-core-dev libignition-common4-dev libignition-common4-events \
    libignition-common4-events-dev libignition-common4-graphics libignition-common4-graphics-dev \
    libignition-common4-profiler libignition-common4-profiler-dev libignition-fuel-tools7 \
    libignition-fuel-tools7-dev libignition-gazebo6 libignition-gazebo6-dev \
    libignition-gazebo6-plugins libignition-gui6 libignition-gui6-dev libignition-math6 \
    libignition-math6-dev libignition-math6-eigen3-dev libignition-msgs8 libignition-msgs8-dev \
    libignition-physics5 libignition-physics5-bullet libignition-physics5-bullet-dev \
    libignition-physics5-core-dev libignition-physics5-dartsim libignition-physics5-dartsim-dev \
    libignition-physics5-dev libignition-physics5-heightmap-dev libignition-physics5-mesh-dev \
    libignition-physics5-sdf-dev libignition-physics5-tpe libignition-physics5-tpe-dev \
    libignition-physics5-tpelib libignition-physics5-tpelib-dev libignition-plugin \
    libignition-plugin-dev libignition-rendering6 libignition-rendering6-core-dev \
    libignition-rendering6-dev libignition-rendering6-ogre1 libignition-rendering6-ogre1-dev \
    libignition-rendering6-ogre2 libignition-rendering6-ogre2-dev libignition-sensors6 \
    libignition-sensors6-air-pressure libignition-sensors6-air-pressure-dev \
    libignition-sensors6-altimeter libignition-sensors6-altimeter-dev \
    libignition-sensors6-boundingbox-camera libignition-sensors6-boundingbox-camera-dev \
    libignition-sensors6-camera libignition-sensors6-camera-dev libignition-sensors6-core-dev \
    libignition-sensors6-depth-camera libignition-sensors6-depth-camera-dev \
    libignition-sensors6-dev libignition-sensors6-force-torque \
    libignition-sensors6-force-torque-dev libignition-sensors6-gpu-lidar \
    libignition-sensors6-gpu-lidar-dev libignition-sensors6-imu libignition-sensors6-imu-dev \
    libignition-sensors6-lidar libignition-sensors6-lidar-dev libignition-sensors6-logical-camera \
    libignition-sensors6-logical-camera-dev libignition-sensors6-magnetometer \
    libignition-sensors6-magnetometer-dev libignition-sensors6-navsat \
    libignition-sensors6-navsat-dev libignition-sensors6-rendering \
    libignition-sensors6-rendering-dev libignition-sensors6-rgbd-camera \
    libignition-sensors6-rgbd-camera-dev libignition-sensors6-segmentation-camera \
    libignition-sensors6-segmentation-camera-dev libignition-sensors6-thermal-camera \
    libignition-sensors6-thermal-camera-dev libignition-tools-dev libignition-transport11 \
    libignition-transport11-core-dev libignition-transport11-dev libignition-transport11-log \
    libignition-transport11-log-dev libignition-transport11-parameters \
    libignition-transport11-parameters-dev libignition-utils1 libignition-utils1-cli-dev \
    libignition-utils1-dev libilmbase-dev libilmbase25 libimagequant0 libip4tc2 libjack-jackd2-0 \
    libjbig-dev libjpeg-dev libjpeg-turbo8-dev libjpeg8-dev libjs-jquery-ui libjson-c5 \
    libjson-glib-1.0-0 libjson-glib-1.0-common libjsoncpp-dev libjxr0 libkadm5clnt-mit12 \
    libkadm5srv-mit12 libkdb5-10 libkmlbase1 libkmldom1 libkmlengine1 libkrb5-dev liblbfgsb0 \
    liblcms2-2 liblept5 liblilv-0-0 libllvm14 liblmdb0 libltdl-dev libmagic-mgc libmagic1 \
    libmaxminddb0 libmbim-glib4 libmbim-proxy libmd-dev libmd4c0 libmfx1 libminizip-dev \
    libminizip1 libmm-glib0 libmount-dev libmp3lame0 libmpg123-0 libmysofa1 libmysqlclient21 \
    libnetcdf19 libnl-3-200 libnl-3-dev libnl-genl-3-200 libnl-route-3-200 libnl-route-3-dev \
    libnorm-dev libnorm1 libnotify4 libnspr4 libnss-mdns libnss-systemd libnss3 libnuma-dev \
    libnuma1 liboctomap-dev liboctomap1.9 libodbc2 libodbcinst2 libode-dev libode8 libogdi4.1 \
    libogg-dev libogg0 libogre-1.9-dev libogre-1.9.0v5 libogre-next-dev libogrenexthlmspbs2.2.5 \
    libogrenexthlmsunlit2.2.5 libogrenextmain2.2.5 libogrenextmeshlodgenerator2.2.5 \
    libogrenextoverlay2.2.5 libogrenextplanarreflections2.2.5 libogrenextsceneformat2.2.5 \
    libopenal-data libopenal1 libopenblas-dev libopenblas-pthread-dev libopenblas0 \
    libopenblas0-pthread libopencv-calib3d-dev libopencv-calib3d4.5d libopencv-contrib-dev \
    libopencv-contrib4.5d libopencv-core-dev libopencv-core4.5d libopencv-dev libopencv-dnn-dev \
    libopencv-dnn4.5d libopencv-features2d-dev libopencv-features2d4.5d libopencv-flann-dev \
    libopencv-flann4.5d libopencv-highgui-dev libopencv-highgui4.5d libopencv-imgcodecs-dev \
    libopencv-imgcodecs4.5d libopencv-imgproc-dev libopencv-imgproc4.5d libopencv-ml-dev \
    libopencv-ml4.5d libopencv-objdetect-dev libopencv-objdetect4.5d libopencv-photo-dev \
    libopencv-photo4.5d libopencv-shape-dev libopencv-shape4.5d libopencv-stitching-dev \
    libopencv-stitching4.5d libopencv-superres-dev libopencv-superres4.5d libopencv-video-dev \
    libopencv-video4.5d libopencv-videoio-dev libopencv-videoio4.5d libopencv-videostab-dev \
    libopencv-videostab4.5d libopencv-viz-dev libopencv-viz4.5d libopencv4.5-java \
    libopencv4.5d-jni libopenexr-dev libopenexr25 libopengl-dev libopengl0 libopenjp2-7 \
    libopenmpi-dev libopenmpi3 libopenmpt0 libopus0 liborc-0.4-0 libpam-cap libpam-systemd \
    libpcre16-3 libpcre2-16-0 libpcre2-32-0 libpcre2-dev libpcre2-posix3 libpcre3-dev libpcre32-3 \
    libpcrecpp0v5 libpcsclite1 libpgm-5.3-0 libpgm-dev libpmix-dev libpmix2 libpng-dev \
    libpng-tools libpocketsphinx3 libpolkit-agent-1-0 libpolkit-gobject-1-0 libpoppler118 \
    libpostproc-dev libpostproc55 libpq5 libproj22 libprotobuf-dev libprotobuf-lite23 \
    libprotobuf23 libprotoc-dev libprotoc23 libproxy1v5 libpsm-infinipath1 libpsm2-2 \
    libpthread-stubs0-dev libpulse0 libpyside2-dev libpyside2-py3-5.15 libqhull-r8.0 libqmi-glib5 \
    libqmi-proxy libqt5charts5 libqt5concurrent5 libqt5core5a libqt5dbus5 libqt5designer5 \
    libqt5gui5 libqt5help5 libqt5location5 libqt5location5-plugins libqt5network5 libqt5opengl5 \
    libqt5opengl5-dev libqt5positioning5 libqt5positioning5-plugins libqt5positioningquick5 \
    libqt5printsupport5 libqt5qml5 libqt5qmlmodels5 libqt5qmlworkerscript5 libqt5quick5 \
    libqt5quickcontrols2-5 libqt5quickparticles5 libqt5quickshapes5 libqt5quicktemplates2-5 \
    libqt5quicktest5 libqt5quickwidgets5 libqt5sql5 libqt5sql5-sqlite libqt5svg5 libqt5test5 \
    libqt5widgets5 libqt5xml5 librabbitmq4 libraqm0 libraw1394-11 libraw1394-dev libraw1394-tools \
    libraw20 librdmacm1 librsvg2-2 librsvg2-common librttopo1 librubberband2 libruby3.0 \
    libsamplerate0 libsdformat12 libsdformat12-dev libsdl2-2.0-0 libselinux1-dev libsepol-dev \
    libserd-0-0 libshiboken2-dev libshiboken2-py3-5.15 libshine3 libsigsegv2 libslang2 libsm-dev \
    libsnappy1v5 libsndfile1 libsndio7.0 libsocket++1 libsodium-dev libsord-0-0 libsoup2.4-1 \
    libsoup2.4-common libsoxr0 libspatialite7 libspeex1 libsphinxbase3 libsratom-0-0 \
    libsrt1.4-gnutls libssh-gcrypt-4 libsuperlu5 libswresample-dev libswresample3 libswscale-dev \
    libswscale5 libsz2 libtbb-dev libtbb12 libtbb2 libtbbmalloc2 libtcl8.6 libtesseract4 \
    libtheora-dev libtheora0 libtiff-dev libtiffxx5 libtk8.6 libtool libtwolame0 libucx0 \
    libudfread0 liburdfdom-dev liburdfdom-headers-dev liburdfdom-model-state3.0 \
    liburdfdom-model3.0 liburdfdom-sensor3.0 liburdfdom-world3.0 liburiparser1 libusb-1.0-0 \
    libva-drm2 libva-x11-2 libva2 libvdpau1 libvidstab1.1 libvisual-0.4-0 libvorbis0a \
    libvorbisenc2 libvorbisfile3 libvpx7 libvtk9.1 libvulkan-dev libvulkan1 libwayland-cursor0 \
    libwayland-egl1 libwebpdemux2 libwebpmux3 libx11-dev libx264-163 libx265-199 libxau-dev \
    libxaw7-dev libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0 libxcb-shape0 \
    libxcb-util1 libxcb-xinerama0 libxcb-xinput0 libxcb-xkb1 libxcb1-dev libxcomposite1 \
    libxdamage1 libxdmcp-dev libxerces-c3.2 libxext-dev libxi6 libxinerama1 libxkbcommon-x11-0 \
    libxkbcommon0 libxml2-dev libxmu-dev libxmu-headers libxnvctrl0 libxpm-dev libxrandr-dev \
    libxrandr2 libxrender-dev libxsimd-dev libxss1 libxt-dev libxtst6 libxv1 libxvidcore4 libzimg2 \
    libzip-dev libzip4 libzmq3-dev libzmq5 libzvbi-common libzvbi0 libzzip-0-13 m4 mesa-va-drivers \
    mesa-vdpau-drivers mesa-vulkan-drivers modemmanager mpi-default-bin mpi-default-dev \
    mysql-common networkd-dispatcher ocl-icd-libopencl1 opencv-data openmpi-bin openmpi-common \
    openssh-client pkexec pocketsphinx-en-us policykit-1 polkitd poppler-data proj-bin proj-data \
    protobuf-compiler pyqt5-dev python-matplotlib-data python3-appdirs python3-beniget \
    python3-brotli python3-cycler python3-decorator python3-fonttools python3-fs python3-gast \
    python3-gi python3-kiwisolver python3-lz4 python3-matplotlib python3-mpmath python3-olefile \
    python3-opencv python3-pil python3-pil.imagetk python3-ply python3-pyqt5 python3-pyqt5.qtsvg \
    python3-pyqt5.sip python3-pyside2.qtcore python3-pyside2.qtgui python3-pyside2.qtsvg \
    python3-pyside2.qtwidgets python3-pythran python3-scipy python3-sip-dev python3-sympy \
    python3-tk python3-ufolib2 python3-unicodedata2 qml-module-qt-labs-folderlistmodel \
    qml-module-qt-labs-platform qml-module-qt-labs-settings qml-module-qtcharts \
    qml-module-qtgraphicaleffects qml-module-qtlocation qml-module-qtpositioning qml-module-qtqml \
    qml-module-qtqml-models2 qml-module-qtquick-controls qml-module-qtquick-controls2 \
    qml-module-qtquick-dialogs qml-module-qtquick-layouts qml-module-qtquick-privatewidgets \
    qml-module-qtquick-templates2 qml-module-qtquick-window2 qml-module-qtquick2 \
    qt5-gtk-platformtheme qt5-qmake qt5-qmake-bin qt5-qmltooling-plugins qtbase5-dev \
    qtbase5-dev-tools qtchooser qtdeclarative5-dev qtdeclarative5-dev-tools qtquickcontrols2-5-dev \
    qttranslations5-l10n rake ros-humble-actuator-msgs ros-humble-compressed-depth-image-transport \
    ros-humble-compressed-image-transport ros-humble-cv-bridge ros-humble-gps-msgs \
    ros-humble-ignition-cmake2-vendor ros-humble-ignition-math6-vendor ros-humble-image-transport \
    ros-humble-image-transport-plugins ros-humble-interactive-markers ros-humble-laser-geometry \
    ros-humble-libcurl-vendor ros-humble-map-msgs ros-humble-python-qt-binding ros-humble-qt-gui \
    ros-humble-qt-gui-cpp ros-humble-qt-gui-py-common ros-humble-resource-retriever \
    ros-humble-ros-gz ros-humble-ros-gz-bridge ros-humble-ros-gz-image \
    ros-humble-ros-gz-interfaces ros-humble-ros-gz-sim ros-humble-ros-gz-sim-demos \
    ros-humble-rqt-gui ros-humble-rqt-gui-cpp ros-humble-rqt-gui-py ros-humble-rqt-image-view \
    ros-humble-rqt-plot ros-humble-rqt-py-common ros-humble-rqt-topic \
    ros-humble-rviz-assimp-vendor ros-humble-rviz-common ros-humble-rviz-default-plugins \
    ros-humble-rviz-ogre-vendor ros-humble-rviz-rendering ros-humble-rviz2 \
    ros-humble-sdformat-urdf ros-humble-tango-icons-vendor ros-humble-theora-image-transport \
    ros-humble-vision-msgs ros-humble-xacro ruby ruby-net-telnet ruby-rubygems ruby-webrick \
    ruby-xmlrpc ruby3.0 rubygems-integration sdformat12-sdf session-migration shared-mime-info \
    shiboken2 sip-dev systemd systemd-sysv systemd-timesyncd tango-icon-theme tcl tcl8.6 \
    tk8.6-blt2.5 ubuntu-mono unicode-data unixodbc-common unzip usb-modeswitch usb-modeswitch-data \
    uuid-dev va-driver-all vdpau-driver-all wpasupplicant x11proto-dev xauth xorg-sgml-doctools \
    xtrans-dev zip \ 
    ca-certificates-java default-jdk default-jdk-headless default-jre default-jre-headless \
    default-libmysqlclient-dev fonts-dejavu-extra freeglut3 hdf5-helpers java-common libaec-dev \
    libaom-dev libarmadillo-dev libarpack2-dev libatk-wrapper-java libatk-wrapper-java-jni \
    libblosc-dev libcfitsio-dev libcfitsio-doc libcharls-dev libdav1d-dev libde265-dev \
    libdouble-conversion-dev libflann-dev libflann1.9 libfontconfig-dev libfontconfig1-dev \
    libfreexl-dev libfyba-dev libgdal-dev libgeos-dev libgeotiff-dev libgif-dev libgl2ps-dev \
    libhdf4-alt-dev libhdf5-cpp-103-1 libhdf5-dev libhdf5-fortran-102 libhdf5-hl-cpp-100 \
    libhdf5-hl-fortran-100 libhdf5-mpi-dev libhdf5-openmpi-103-1 libhdf5-openmpi-cpp-103-1 \
    libhdf5-openmpi-dev libhdf5-openmpi-fortran-102 libhdf5-openmpi-hl-100 \
    libhdf5-openmpi-hl-cpp-100 libhdf5-openmpi-hl-fortran-100 libheif-dev libhyphen0 libjson-c-dev \
    libkml-dev libkmlconvenience1 libkmlregionator1 libkmlxsd1 liblz4-dev libmysqlclient-dev \
    libnetcdf-c++4 libnetcdf-cxx-legacy-dev libnetcdf-dev libodbccr2 libogdi-dev libopenjp2-7-dev \
    libopenni-dev libopenni-sensor-pointclouds0 libopenni0 libopenni2-0 libopenni2-dev libpcap0.8 \
    libpcl-apps1.12 libpcl-common1.12 libpcl-dev libpcl-features1.12 libpcl-filters1.12 \
    libpcl-io1.12 libpcl-kdtree1.12 libpcl-keypoints1.12 libpcl-ml1.12 libpcl-octree1.12 \
    libpcl-outofcore1.12 libpcl-people1.12 libpcl-recognition1.12 libpcl-registration1.12 \
    libpcl-sample-consensus1.12 libpcl-search1.12 libpcl-segmentation1.12 libpcl-stereo1.12 \
    libpcl-surface1.12 libpcl-tracking1.12 libpcl-visualization1.12 libpoppler-dev \
    libpoppler-private-dev libpq-dev libproj-dev libqt5designercomponents5 libqt5sensors5 \
    libqt5webchannel5 libqt5webkit5 libqt5webkit5-dev librttopo-dev libspatialite-dev \
    libsuperlu-dev liburiparser-dev libusb-1.0-0-dev libusb-1.0-doc libutfcpp-dev libvtk9-dev \
    libvtk9-java libvtk9-qt-dev libvtk9.1-qt libwebp-dev libwoff1 libx265-dev libxerces-c-dev \
    libxft-dev libxss-dev libxxf86dga1 openjdk-11-jdk openjdk-11-jdk-headless openjdk-11-jre \
    openjdk-11-jre-headless openni-utils python3-mpi4py python3-vtk9 qdoc-qt5 qhelpgenerator-qt5 \
    qt5-assistant qtattributionsscanner-qt5 qttools5-dev qttools5-dev-tools qttools5-private-dev \
    ros-humble-pcl-conversions ros-humble-pcl-msgs tcl-dev tcl8.6-dev tk tk-dev tk8.6 tk8.6-dev \
    unixodbc-dev vtk9 x11-utils xbitmaps xterm

RUN mkdir /workspace
WORKDIR /workspace

FROM base AS dev

RUN apt update && apt install -y --no-install-recommends \
    vim \
    tmux \
    x11-apps

CMD ["/bin/bash"]

