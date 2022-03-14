import joblib

model = joblib.load('model.joblib')


def predictChurn(creditScore, country, gender, age, tenure, balance, numProducts, hasCrCard, isActive, salary):

	genderDict = {'male':1, 'female':0}
	countryDict = {'germany':0, 'spain':0}
	#['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Germany', 'Spain', 'Male']

	if country.lower() == 'germany':
		countryDict['germany'] = 1
	elif country.lower() == 'spain':
		countryDict['spain'] = 1
	else: #If france or something else
		pass


	if gender.lower() == 'male':
		genderResult = genderDict['male']
	else:
		genderResult = genderDict['female']


	customer_data = [[creditScore, age, tenure, balance, numProducts, hasCrCard, isActive, salary, countryDict['germany'], countryDict['spain'], genderResult]]
	pred = model.predict(customer_data)

	if pred[0] == 0:
		return '\n*********************************\n\nCustomer is less likely to leave.'
	elif pred[0] == 1:
		return "\n*********************************\n\nCustomer is likely to leave."



if __name__ == '__main__':
	creditScore = int(input("Enter CreditScore: "))
	country = input("Enter Country (germany/spain): ")
	gender = input("Enter Gender: ")
	age = int(input("Enter Age: "))
	tenure = int(input("Enter tenure: "))
	balance = float(input("Enter balance: "))
	numProducts = int(input("Enter Number of Products: "))
	hasCrCard = input("Has credit Card? (1 for yes / 0 for no): ")
	isActive = input("Active Member? (1 for yes / 0 for no): ")
	salary = float(input("Enter salary: "))

	#Call Method to predict
	result = predictChurn(creditScore, country, gender, age, tenure, balance, numProducts, hasCrCard, isActive, salary)
	print(result)
