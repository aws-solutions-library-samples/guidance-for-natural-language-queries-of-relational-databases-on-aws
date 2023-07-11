## Generative AI-enabled Natural Language Relational Database Queries

Natural Language Query (NLQ) demonstration using Amazon RDS for PostgreSQL and Amazon SageMaker JumpStart Foundation Models and optionally OpenAI's models via their API.

## Foundation Model Choice and Accuracy of NQL

The selection of the Foundation Model (FM) for NLQ plays a crucial role in the application's ability to accurately translate natural language questions into natural language answers; not all FMs possess this capability. Additionally, the accuracy of the NLQ process relies heavily on the chosen model, as well as other factors such as the prompt, prompt-template, and sample queries used for in-context learning (also known as few-shot prompting).

OpenAI GPT-3 and GPT-4 series models, including `text-davinci-003` (Legacy), `gpt-3.5-turbo`, and the latest addition, `gpt-4`. These models are considered state-of-the-art for NLQ, providing highly accurate responses to a wide range of complex NLQ questions, with minimal in-context learning. As an alternative to OpenAI, models such as the `google/flan-t5-xxl` and `google/flan-t5-xxl-fp16` models available through Amazon SageMaker JumpStart Foundation Models. It's important to note that while the `google/flan-t5` series of models are a popular choice, their capabilities for NLQ are only a fraction of what OpenAI's GPT-3 and GPT-4 series models offer. The `google/flan-t5-xxl-fp16` model may fail to return an answer, provide incorrect answers, or cause the JumpStart Foundation Model to experience timeouts when faced with even moderately complex questions.

Transitioning from Amazon SageMaker JumpStart Foundation Models to OpenAI's models via their API eliminates the need for the deployment of the `NlqSageMakerEndpointStack` CloudFormation stack. If the stack has already been deployed, it can be deleted. The next build the Amazon ECR Docker Image using the `Dockerfile_OpenAI` Dockerfile and push it to the ECR repository. Finally, deploy the `NlqEcsOpenAIStack.yaml` CloudFormation template file. To utilize OpenAI's models, you will also need to create an OpenAI account and obtain your personal secret API key.

You can also replace the `google/flan-t5-xxl-fp16` JumpStart Foundation Model hosted by the Amazon SageMaker endpoint, deployed by the existing `NlqSageMakerEndpointStack.yaml` CloudFormation template file. You will first need to modify the `NlqSageMakerEndpointStack.yaml` CloudFormation template file and update the deployed CloudFormation stack, `NlqSageMakerEndpointStack`. Additionally, you will have to make adjustments to the `app_sagemaker.py` file by modifying the `ContentHandler` Class to match the response payload of the chosen model. Rebuilding the Amazon ECR Docker Image using the `Dockerfile_SageMaker` Dockerfile and pushing it to the ECR repository will also be necessary. Lastly, you should update the deployed ECS task and service, which are part of the `NlqEcsStackSageMaker.yaml` CloudFormation template file (`NlqEcsStackSageMaker` CloudFormation stack).

## Instructions

1. If using the Amazon SageMaker JumpStart Foundation Models option, make you have the required EC2 instance for the endpoint inference, or request it using Service Quotas in the AWS Management Console (e.g., `ml.g5.24xlarge` for the `google/flan-t5-xxl-fp16` model: https://us-east-1.console.aws.amazon.com/servicequotas/home/services/sagemaker/quotas/L-6821867B).
2. Create the required secrets in AWS Secret Manager using the AWS CLI.
3. Deploy the `NlqMainStack` CloudFormation template.
4. Build and push the `nlq-genai:1.0.0-sm` Docker image to the new Amazon ECR repository, or alternately the the `nlq-genai:1.0.0-oai` Docker image for use with Option 2: OpenAI API.
5. Import the included sample data into the Amazon RDS MoMA database.
6. Add the `nlqapp` user to the MoMA database.
7. Deploy the `NlqSageMakerEndpointStack` CloudFormation template, using the Amazon SageMaker JumpStart Foundation Models option.
8. Deploy the `NlqEcsSageMakerStack` CloudFormation template, or alternately the `NlqOpenAIStack` CloudFormation template for use with Option 2: OpenAI API.

## 2. Create AWS Secret Manager Secrets

Make sure you update usernames and password.

```sh
aws secretsmanager create-secret \
    --name /nlq/MasterUsername \
    --description "Master username for RDS instance." \
    --secret-string <your_master_username>

aws secretsmanager create-secret \
    --name /nlq/NLQAppUsername \
    --description "Master username for RDS instance." \
    --secret-string <your_nqlapp_username>

aws secretsmanager create-secret \
    --name /nlq/NLQAppUserPassword \
    --description "Master username for RDS instance." \
    --secret-string "<your_nqlapp_password>"

# optional for Option 2: OpenAI API/model
aws secretsmanager create-secret \
    --name /nlq/OpenAIAPIKey \
    --description "OpenAI API key." \
    --secret-string "<your_openai_api_key"
```

## 3. Deploy the Main NLQ Stack: Networking, Security, RDS Instance, and ECR Repository

```sh
cd cloudformation/

aws cloudformation create-stack \
  --stack-name NlqMainStack \
  --template-body file://NlqMainStack.yaml \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameters \
  		ParameterKey="MyIpAddress",ParameterValue=$(curl -s http://checkip.amazonaws.com/)/32
```

## 4. Build and Push the Docker Image to ECR

You can build the image locally, in a CI/CD pipeline, using SageMaker Notebook environment, or AWS Cloud9.

```sh
# Located in the output from the nlq-genai-infra CloudFormation template
cd docker/

# e.g. 111222333444.dkr.ecr.us-east-1.amazonaws.com/nlq-genai
ECS_REPOSITORY="<you_ecs_repository>"
```

Option 1: SageMaker JumpStart FM Endpoint

```sh
TAG="1.0.0-sm"

aws ecr get-login-password --region us-east-1 | \
	docker login --username AWS --password-stdin $ECS_REPOSITORY

docker build -f Dockerfile_SageMaker -t $ECS_REPOSITORY:$TAG .

docker push $ECS_REPOSITORY:$TAG
```

Option 2: OpenAI API

```sh
TAG="1.0.0-oai"

docker build -f Dockerfile_OpenAI -t $ECS_REPOSITORY:$TAG .

docker push $ECS_REPOSITORY:$TAG
```

## 5. Import Sample Data and Configure the MoMA Database

1. Connect to the `moma` database using you preferred PostgreSQL tool. You may need to enable `Public access` for the RDS instance temporarily depending on how you connect to the database.
2. Create the two MoMA collection tables into the `moma` database.

```sql
CREATE TABLE public.artists (
	artist_id integer NOT NULL,
	full_name character varying(200),
	nationality character varying(50),
	gender character varying(25),
	birth_year integer,
	death_year integer,
	CONSTRAINT artists_pk PRIMARY KEY (artist_id)
)

CREATE TABLE public.artworks (
	artwork_id integer NOT NULL,
	title character varying(500),
	artist_id integer NOT NULL,
	date integer,
	medium character varying(250),
	dimensions text,
	acquisition_date text,
	credit text,
	catalogue character varying(250),
	department character varying(250),
	classification character varying(250),
	object_number text,
	diameter_cm text,
	circumference_cm text,
	height_cm text,
	length_cm text,
	width_cm text,
	depth_cm text,
	weight_kg text,
	durations integer,
	CONSTRAINT artworks_pk PRIMARY KEY (artwork_id)
)
```

3. Import the data into the `moma` database using the text files in the `/data` sub-directory. The data contains a header row and pipe-delimited ('|').

```txt
# examples commands from pgAdmin4
--command " "\\copy public.artists (artist_id, full_name, nationality, gender, birth_year, death_year) FROM 'moma_public_artists.txt' DELIMITER '|' CSV HEADER QUOTE '\"' ESCAPE '''';""

--command " "\\copy public.artworks (artwork_id, title, artist_id, date, medium, dimensions, acquisition_date, credit, catalogue, department, classification, object_number, diameter_cm, circumference_cm, height_cm, length_cm, width_cm, depth_cm, weight_kg, durations) FROM 'moma_public_artworks.txt' DELIMITER '|' CSV HEADER QUOTE '\"' ESCAPE '''';""
```

## 6. Add NLP Application to the MoMA Database

Create the read-only NQL Application database user account. Update the username and password you configured in step 2, with the secrets.

```sql
CREATE ROLE <your_nqlapp_username> WITH
	LOGIN
	NOSUPERUSER
	NOCREATEDB
	NOCREATEROLE
	INHERIT
	NOREPLICATION
	CONNECTION LIMIT -1
	PASSWORD '<your_nqlapp_password>';

GRANT pg_read_all_data TO <your_nqlapp_username>;
```

## 7. Deploy the ML Stack: Model, Endpoint

Option 1: SageMaker JumpStart FM Endpoint

```sh
cd cloudformation/

aws cloudformation create-stack \
  --stack-name NlqSageMakerEndpointStack \
  --template-body file://NlqSageMakerEndpointStack.yaml \
  --capabilities CAPABILITY_NAMED_IAM
```

## 8. Deploy the ECS Service Stack: Task, Service

Option 1: SageMaker JumpStart FM Endpoint

```sh
aws cloudformation create-stack \
  --stack-name NlqEcsSageMakerStack \
  --template-body file://NlqEcsSageMakerStack.yaml \
  --capabilities CAPABILITY_NAMED_IAM
```

Option 2: OpenAI API

```sh
aws cloudformation create-stack \
  --stack-name NlqEcsOpenAIStack \
  --template-body file://NlqEcsOpenAIStack.yaml \
  --capabilities CAPABILITY_NAMED_IAM
```

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

````
