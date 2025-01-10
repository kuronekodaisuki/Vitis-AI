#include "Object.h"

/// <summary>
/// JSONize
/// </summary>
/// <param name="stream"></param>
/// <returns></returns>
bool Object::Send(std::ostream& stream)
{
	char buffer[120];
	//if (_names == nullptr)
		sprintf(buffer, "{\"x\":%f,\"y\":%f,\"width\":%f,\"height\":%f,\"confidence\":%f,\"label\":%d}", rect.x, rect.y, rect.width, rect.height, prob, label);
	//else
	//	sprintf(buffer, "{\"x\":%f,\"y\":%f,\"width\":%f,\"height\":%f,\"confidence\":%f,\"label\":%s}", rect.x, rect.y, rect.width, rect.height, prob, _names[label]);
	stream << buffer;
	return true;
}

/*
void Object::Set(char* names[])
{
	_names = names; 
}
*/