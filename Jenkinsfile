pipeline {
    agent any
    parameters {
        choice(name: 'VERSION', choices: ['1.0', '2.0', '3.0'], description: 'Select the version to build')
        booleanParam(name: 'executeTests', defaultValue: true, description: 'Run tests after build')
    }
    stages {

        stage("build") {
            
            steps {
                echo "Building..."
            }

        }

        stage("test") {
            when {
                expression {
                    BRANCH_NAME == 'main' && params.executeTests == true
                }
            }
            steps {
                echo "Testing..."
            }

        }

        stage("deploy") {

            steps {
                echo "Deploying..."
                echo "Deployed version: ${params.VERSION}"
            }

        }
        
    }

    post {
        always {
            echo "Pipeline finished."
        }
        success {
            echo "Pipeline succeeded."
        }
        failure {
            echo "Pipeline failed."
        }
    }
}