# Web2Text

Web2Text needs Tensorflow 1.x and Scala with Java < 11. To match the requirements, create a Python 3.7 venv named `venv` in this directory:

    python3.7 -m venv
    source venv/bin/activate

Then run

    pip install numpy==1.18.0 tensorflow==1.15.0 tensorflow-gpu==1.15.0 protobuf==3.20.1 future==0.18.3

The Web2Text JAR is already included in this repository, but can be rebuilt from the webtext sources. Add the following lines to `build.sbt`:

    ThisBuild / assemblyMergeStrategy := {
        case PathList("META-INF", xs@_*) => MergeStrategy.discard
        case x => MergeStrategy.first
    }

and create the file `project/plugins.sbt` with the following line:

    'addSbtPlugin("com.eed3si9n" % "sbt-assembly" % "2.1.1")

Then build the fat JAR with:

    sbt -java-home /usr/lib/jvm/java-8-openjdk-amd64 assembly

Further build and run instructions: https://web.archive.org/web/20211016233421/https://xaviergeerinck.com/post/ai/web2text/
