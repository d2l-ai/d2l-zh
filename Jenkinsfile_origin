stage("Build and Publish") {
  // such as d2l-en and d2l-zh
  def REPO_NAME = env.JOB_NAME.split('/')[0]
  // such as en and zh
  def LANG = REPO_NAME.split('-')[1]
  // The current branch or the branch this PR will merge into
  def TARGET_BRANCH = env.CHANGE_TARGET ? env.CHANGE_TARGET : env.BRANCH_NAME
  // such as d2l-en-master
  def TASK = REPO_NAME + '-' + TARGET_BRANCH
  node {
    ws("workspace/${TASK}") {
      checkout scm
      // conda environment
      def ENV_NAME = "${TASK}-${EXECUTOR_NUMBER}";

      sh label: "Build Environment", script: """set -ex
      conda env update -n ${ENV_NAME} -f static/build.yml
      conda activate ${ENV_NAME}
      pip uninstall -y d2lbook
      pip install git+https://github.com/d2l-ai/d2l-book
      pip list
      nvidia-smi
      """

      sh label: "Sanity Check", script: """set -ex
      conda activate ${ENV_NAME}
      d2lbook build outputcheck tabcheck
      """

      sh label: "Execute Notebooks", script: """set -ex
      conda activate ${ENV_NAME}
      ./static/cache.sh restore _build/eval/data
      d2lbook build eval
      ./static/cache.sh store _build/eval/data
      """

      sh label: "Execute Notebooks [PyTorch]", script: """set -ex
      conda activate ${ENV_NAME}
      ./static/cache.sh restore _build/eval_pytorch/data
      d2lbook build eval --tab pytorch
      d2lbook build slides --tab pytorch
      ./static/cache.sh store _build/eval_pytorch/data
      """

      sh label: "Execute Notebooks [TensorFlow]", script: """set -ex
      conda activate ${ENV_NAME}
      ./static/cache.sh restore _build/eval_tensorflow/data
      export TF_CPP_MIN_LOG_LEVEL=3
      d2lbook build eval --tab tensorflow
      ./static/cache.sh store _build/eval_tensorflow/data
      """

      sh label:"Build HTML", script:"""set -ex
      conda activate ${ENV_NAME}
      ./static/build_html.sh
      """

      sh label:"Build PDF", script:"""set -ex
      conda activate ${ENV_NAME}
      d2lbook build pdf
      """

      if (env.BRANCH_NAME == 'release') {
        sh label:"Release", script:"""set -ex
        conda activate ${ENV_NAME}
        d2lbook build pkg
        d2lbook deploy html pdf pkg colab sagemaker slides --s3 s3://en.d2l.ai/
        """

        sh label:"Release d2l", script:"""set -ex
        conda activate ${ENV_NAME}
        pip install setuptools wheel twine
        python setup.py bdist_wheel
        # twine upload dist/*
        """
      } else {
        sh label:"Publish", script:"""set -ex
        conda activate ${ENV_NAME}
        d2lbook deploy html pdf slides --s3 s3://preview.d2l.ai/${JOB_NAME}/
        """
        if (env.BRANCH_NAME.startsWith("PR-")) {
            pullRequest.comment("Job ${JOB_NAME}/${BUILD_NUMBER} is complete. \nCheck the results at http://preview.d2l.ai/${JOB_NAME}/")
        }
      }
    }
  }
}
