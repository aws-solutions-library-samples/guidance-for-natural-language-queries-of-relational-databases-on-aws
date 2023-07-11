## Generative AI-enabled Natural Language Relational Database Queries

Natural Language Query (NLQ) demonstration using Amazon RDS for PostgreSQL and Amazon SageMaker JumpStart Foundation Models and optionally OpenAI's models via their API.

## Foundation Models and Accuracy of NQL

The Foundation Model (aka Large Foundation Model (LFM) or Language Model (LLM)) selected for NLQ will have a critical impact on the ability of the application to translate natural language questions into natural language answers. Only certain FMs/LLMs have this capability. Further, even with a model that supports NLQ, the accuracy of the NLQ process is also heavily dependent on the model, along with other variables such as the prompt, prompt-template, and sample queries passed as part of the in-context learning (aka few-shot prompting).

OpenAI's GPT-3 and GPT-4 series `text-davinci-003` (Legacy), `gpt-3.5-turbo`, and most recently `gpt-4` were used to develop the NLQ application. These models represent state-of-the-art for NLQ. They can respond to a wide range of complex NLQ questions accurately. This solution offer the option to substitute the `google/flan-t5-xxl` model, available through Amazon SageMaker JumpStart Foundation Models with OpenAI's models via their API. Although the the `google/flan-t5-xxl` model is a widely-used LFM, its capabilities for NLQ pale in comparison to OpenAI's GPT-3 series. Answers to even moderately complex questions may be fail, be incorrect, or cause the JumpStart Foundation Model to timeout. You will need to establish an OpenAI account and get your obtain your `OPENAI_API_KEY`.

To change from Amazon SageMaker JumpStart Foundation Models to using OpenAI's models via their API, you will no longer need the the CloudFormation stack, `NlqSageMakerEndpointStack`. This stack can be ignored or deleted if already deployed. Next, you will have to build the Amazon ECR Docker Image using the `Dockerfile_OpenAI` Dockerfile and push to the ECR repository. Finally, deployed the `NlqEcsOpenAIStack.yaml` CloudFormation template file (`NlqEcsOpenAIStack` CloudFormation stack).

To change the JumpStart Foundation Model hosted by the Amazon SageMaker endpoint, first, modify and the `NlqSageMakerEndpointStack.yaml` CloudFormation template file and update the deployed CloudFormation stack, `NlqSageMakerEndpointStack`. You will also have to modify the `app_sagemaker.py` file, adjusting the `ContentHandler` Class to reflect the response payload of the model you select. Next, this will require you to rebuild the Amazon ECR Docker Image using the `Dockerfile_SageMaker` Dockerfile and push to the ECR repository. Lastly, update the deployed ECS task and service, which is part of the `NlqEcsStackSageMaker.yaml` CloudFormation template file (`NlqEcsStackSageMaker` CloudFormation stack).

## Instructions

1. If using the Amazon SageMaker JumpStart Foundation Models option, make you have the required EC2 instance for the endpoint inference, or request it using Service Quotas in the AWS Management Console (e.g., `ml.g5.24xlarge`: https://us-east-1.console.aws.amazon.com/servicequotas/home/services/sagemaker/quotas/L-6821867B).
2. Create the required secrets in AWS Secret Manager using the AWS CLI.
3. Deploy the `NlqMainStack` CloudFormation template.
4. Build and push the `nlq-genai:1.0.0-sm` Docker image to the new Amazon ECR repository, or alternately the the `nlq-genai:1.0.0-gai` Docker image for use with Option 2: OpenAI API.
5. Import the included sample data into the Amazon RDS MoMA database.
6. Add the `nlqapp` user to the MoMA database.
7. Deploy the `NlqSageMakerEndpointStack` CloudFormation template.
8. Deploy the `NlqEcsSageMakerStack` CloudFormation template, or alternately the `NlqOpenAIStack` CloudFormation template for use with Option 2: OpenAI API.

## Create AWS Secret Manager Secrets

Make sure you update usernames and password.

```sh
aws secretsmanager create-secret \
    --name /nlq/MasterUsername \
    --description "Master username for RDS instance." \
    --secret-string "postgres"

aws secretsmanager create-secret \
    --name /nlq/NLQAppUsername \
    --description "Master username for RDS instance." \
    --secret-string "nlqapp"

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

## Deploy the Main NLQ Stack: Networking, Security, RDS Instance, and ECR Repository

```sh
cd cloudformation/

aws cloudformation create-stack \
  --stack-name NlqMainStack \
  --template-body file://NlqMainStack.yaml \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameters \
  		ParameterKey="MyIpAddress",ParameterValue=$(curl -s http://checkip.amazonaws.com/)/32
```

## Import Sample Data and Configure the MoMA Database

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

4. Create the NQL Application database user account. Use the `nlqapp` username and password you configured in the `NlqMainStack` CloudFormation template. Make sure you update the password.

```sql
CREATE ROLE nlqapp WITH
	LOGIN
	NOSUPERUSER
	NOCREATEDB
	NOCREATEROLE
	INHERIT
	NOREPLICATION
	CONNECTION LIMIT -1
	PASSWORD '<your_nqlapp_password>';

GRANT pg_read_all_data TO nlqapp;
```

## Build and Push the Docker Image to ECR

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

## Deploy the ML Stack: Model, Endpoint

Option 1: SageMaker JumpStart FM Endpoint

```sh
cd cloudformation/

aws cloudformation create-stack \
  --stack-name NlqSageMakerEndpointStack \
  --template-body file://NlqSageMakerEndpointStack.yaml \
  --capabilities CAPABILITY_NAMED_IAM
```

## Deploy the ECS Service Stack: Task, Service

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
